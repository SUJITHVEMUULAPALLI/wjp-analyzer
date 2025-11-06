"""
FastAPI Application
==================

RESTful API for WJP ANALYSER with authentication, rate limiting, and comprehensive endpoints.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import os
import uuid

from ..auth.auth_manager import AuthManager
from ..auth.rbac import Permission, Role
from ..auth.rate_limiter import RateLimiter, RateLimitConfig
from ..auth.audit_logger import SecurityAuditLogger
from ..database import get_db_session
from ..database.models import User, Project, Analysis, Conversion, Nesting
from ..config.config_manager import get_config
from ..tasks import analyze_dxf_task, optimize_nesting_task

logger = logging.getLogger(__name__)

# Initialize components
auth_manager = AuthManager()
rate_limiter = RateLimiter()
audit_logger = SecurityAuditLogger()
config = get_config()

# Create FastAPI app
app = FastAPI(
    title="WJP ANALYSER API",
    description="RESTful API for Waterjet Cutting Analysis System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.cors_origins if hasattr(config.security, 'cors_origins') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure properly in production
)


# Pydantic models
class UserCreate(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    role: str = Field(default="user", description="User role")


class UserLogin(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    id: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]


class ProjectCreate(BaseModel):
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime


class AnalysisCreate(BaseModel):
    project_id: Optional[str] = Field(None, description="Project ID")
    material: str = Field(default="steel", description="Material type")
    thickness: float = Field(default=6.0, description="Material thickness")
    sheet_width: float = Field(default=3000.0, description="Sheet width")
    sheet_height: float = Field(default=1500.0, description="Sheet height")


class AnalysisResponse(BaseModel):
    id: str
    dxf_path: str
    status: str
    processing_time: Optional[float]
    total_length: Optional[float]
    total_area: Optional[float]
    pierce_points: Optional[int]
    estimated_cost: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]


class ConversionCreate(BaseModel):
    project_id: Optional[str] = Field(None, description="Project ID")
    binary_threshold: int = Field(default=180, description="Binary threshold")
    min_area: int = Field(default=500, description="Minimum area")
    dxf_size: float = Field(default=1000.0, description="DXF size")
    simplify_tolerance: float = Field(default=1.0, description="Simplify tolerance")


class ConversionResponse(BaseModel):
    id: str
    image_path: str
    dxf_path: Optional[str]
    preview_path: Optional[str]
    status: str
    processing_time: Optional[float]
    polygons_count: Optional[int]
    contours_count: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]


class NestingCreate(BaseModel):
    project_id: Optional[str] = Field(None, description="Project ID")
    name: str = Field(..., description="Nesting name")
    sheet_width: float = Field(default=3000.0, description="Sheet width")
    sheet_height: float = Field(default=1500.0, description="Sheet height")
    spacing: float = Field(default=10.0, description="Spacing")
    material: str = Field(default="steel", description="Material type")
    thickness: float = Field(default=6.0, description="Material thickness")


class NestingResponse(BaseModel):
    id: str
    name: str
    status: str
    processing_time: Optional[float]
    utilization_rate: Optional[float]
    parts_count: Optional[int]
    sheets_count: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]


class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user."""
    try:
        token_payload = auth_manager.verify_token(credentials.credentials)
        if not token_payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        with get_db_session() as session:
            user = session.query(User).filter(User.id == token_payload.user_id).first()
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            return user
            
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def check_permission(permission: Permission, user: User = Depends(get_current_user)) -> User:
    """Check if user has required permission."""
    if not auth_manager.check_permission(user.id, permission):
        audit_logger.log_permission_denied(
            user_id=user.id,
            ip_address="127.0.0.1",  # Would get from request in production
            user_agent="api",
            resource="api",
            action="access",
            required_permission=permission.value
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {permission.value}"
        )
    return user


async def check_rate_limit(request: Request) -> None:
    """Check rate limit for request."""
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")
    
    rate_limit_info = rate_limiter.check_rate_limit(client_ip)
    
    if not rate_limit_info.allowed:
        audit_logger.log_rate_limit_exceeded(client_ip, user_agent, str(request.url))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(rate_limit_info.limit),
                "X-RateLimit-Remaining": str(rate_limit_info.remaining),
                "X-RateLimit-Reset": str(rate_limit_info.reset_time),
                "Retry-After": str(rate_limit_info.retry_after)
            }
        )


# API Routes

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@app.post("/api/v1/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(user_data: UserCreate, request: Request):
    """Register a new user."""
    await check_rate_limit(request)
    
    try:
        # Validate role
        try:
            role = Role(user_data.role)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {user_data.role}"
            )
        
        # Register user
        user_data_dict = user_data.dict()
        user_data_dict['role'] = role
        user = auth_manager.register_user(**user_data_dict)
        
        audit_logger.log_event(
            audit_logger.SecurityEvent(
                event_type=audit_logger.SecurityEventType.LOGIN_SUCCESS,
                user_id=user['user_id'],
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", "unknown"),
                resource="user_registration",
                action="register",
                success=True,
                details={"message": "User registered successfully"},
                timestamp=datetime.utcnow()
            )
        )
        
        return UserResponse(
            id=user['user_id'],
            email=user['email'],
            first_name=user['first_name'],
            last_name=user['last_name'],
            role=user['role'].value,
            is_active=user['is_active'],
            created_at=user['created_at'],
            last_login=user['last_login']
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post("/api/v1/auth/login", tags=["Authentication"])
async def login_user(login_data: UserLogin, request: Request):
    """Login user and return tokens."""
    await check_rate_limit(request)
    
    try:
        result = auth_manager.authenticate_user(
            email=login_data.email,
            password=login_data.password,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", "unknown")
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        return {
            "access_token": result['access_token'],
            "refresh_token": result['refresh_token'],
            "token_type": "bearer",
            "user": result['user']
        }
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login failed"
        )


@app.post("/api/v1/auth/logout", tags=["Authentication"])
async def logout_user(request: Request, user: User = Depends(get_current_user)):
    """Logout user."""
    await check_rate_limit(request)
    
    # In a real implementation, you would invalidate the token
    # For now, we'll just log the logout
    audit_logger.log_event(
        audit_logger.SecurityEvent(
            event_type=audit_logger.SecurityEventType.LOGOUT,
            user_id=user.id,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", "unknown"),
            resource="authentication",
            action="logout",
            success=True,
            details={"message": "User logged out successfully"},
            timestamp=datetime.utcnow()
        )
    )
    
    return {"message": "Logged out successfully"}


@app.get("/api/v1/projects", response_model=List[ProjectResponse], tags=["Projects"])
async def get_projects(
    request: Request,
    user: User = Depends(lambda: check_permission(Permission.READ_PROJECT))
):
    """Get user's projects."""
    await check_rate_limit(request)
    
    with get_db_session() as session:
        projects = session.query(Project).filter(
            Project.user_id == user.id,
            Project.status == 'active'
        ).all()
        
        return [
            ProjectResponse(
                id=project.id,
                name=project.name,
                description=project.description,
                status=project.status,
                created_at=project.created_at,
                updated_at=project.updated_at
            )
            for project in projects
        ]


@app.post("/api/v1/projects", response_model=ProjectResponse, tags=["Projects"])
async def create_project(
    project_data: ProjectCreate,
    request: Request,
    user: User = Depends(lambda: check_permission(Permission.CREATE_PROJECT))
):
    """Create a new project."""
    await check_rate_limit(request)
    
    with get_db_session() as session:
        project = Project(
            user_id=user.id,
            name=project_data.name,
            description=project_data.description
        )
        session.add(project)
        session.commit()
        session.refresh(project)
        
        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            status=project.status,
            created_at=project.created_at,
            updated_at=project.updated_at
        )


@app.post("/api/v1/projects/{project_id}/analyze", response_model=TaskResponse, tags=["Analysis"])
async def analyze_dxf_endpoint(
    project_id: str,
    analysis_data: AnalysisCreate,
    request: Request,
    dxf_file: UploadFile = File(...),
    user: User = Depends(lambda: check_permission(Permission.ANALYZE_DXF))
):
    """Analyze a DXF file."""
    await check_rate_limit(request)
    
    # Validate file type
    if not dxf_file.filename.endswith('.dxf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a DXF file"
        )
    
    # Save uploaded file
    upload_dir = f"uploads/{user.id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(upload_dir, f"{file_id}.dxf")
    
    with open(file_path, "wb") as buffer:
        content = await dxf_file.read()
        buffer.write(content)
    
    # Create analysis record
    with get_db_session() as session:
        analysis = Analysis(
            user_id=user.id,
            project_id=project_id,
            dxf_path=file_path,
            status='pending'
        )
        session.add(analysis)
        session.commit()
        session.refresh(analysis)
    
    # Start background task
    task = analyze_dxf_task.delay(
        analysis_id=analysis.id,
        dxf_path=file_path,
        user_id=user.id,
        analysis_params=analysis_data.dict()
    )
    
    # Log file upload
    audit_logger.log_file_upload(
        user_id=user.id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent", "unknown"),
        filename=dxf_file.filename,
        file_size=len(content),
        file_type="dxf"
    )
    
    return TaskResponse(
        task_id=task.id,
        status="pending",
        result={"analysis_id": analysis.id}
    )


@app.post("/api/v1/projects/{project_id}/convert", response_model=TaskResponse, tags=["Conversion"])
async def convert_image_endpoint(
    project_id: str,
    conversion_data: ConversionCreate,
    request: Request,
    image_file: UploadFile = File(...),
    user: User = Depends(lambda: check_permission(Permission.CONVERT_IMAGE))
):
    """Convert an image to DXF."""
    await check_rate_limit(request)
    
    # Validate file type
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    if not any(image_file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File must be one of: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file
    upload_dir = f"uploads/{user.id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(upload_dir, f"{file_id}_{image_file.filename}")
    
    with open(file_path, "wb") as buffer:
        content = await image_file.read()
        buffer.write(content)
    
    # Create conversion record
    with get_db_session() as session:
        conversion = Conversion(
            user_id=user.id,
            project_id=project_id,
            image_path=file_path,
            status='pending'
        )
        session.add(conversion)
        session.commit()
        session.refresh(conversion)
    
    # Image conversion is no longer available
    return jsonify({'error': 'Image conversion feature has been removed'}), 404
    
    # Log file upload
    audit_logger.log_file_upload(
        user_id=user.id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent", "unknown"),
        filename=image_file.filename,
        file_size=len(content),
        file_type="image"
    )
    
    return TaskResponse(
        task_id=task.id,
        status="pending",
        result={"conversion_id": conversion.id}
    )


@app.post("/api/v1/nesting/optimize", response_model=TaskResponse, tags=["Nesting"])
async def optimize_nesting_endpoint(
    nesting_data: NestingCreate,
    request: Request,
    dxf_file: UploadFile = File(...),
    user: User = Depends(lambda: check_permission(Permission.CREATE_NESTING))
):
    """Optimize nesting for a DXF file."""
    await check_rate_limit(request)
    
    # Validate file type
    if not dxf_file.filename.endswith('.dxf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a DXF file"
        )
    
    # Save uploaded file
    upload_dir = f"uploads/{user.id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(upload_dir, f"{file_id}.dxf")
    
    with open(file_path, "wb") as buffer:
        content = await dxf_file.read()
        buffer.write(content)
    
    # Create nesting record
    with get_db_session() as session:
        nesting = Nesting(
            user_id=user.id,
            project_id=nesting_data.project_id,
            name=nesting_data.name,
            status='pending'
        )
        session.add(nesting)
        session.commit()
        session.refresh(nesting)
    
    # Start background task
    task = optimize_nesting_task.delay(
        nesting_id=nesting.id,
        input_dxf_path=file_path,
        user_id=user.id,
        nesting_params=nesting_data.dict()
    )
    
    # Log file upload
    audit_logger.log_file_upload(
        user_id=user.id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent", "unknown"),
        filename=dxf_file.filename,
        file_size=len(content),
        file_type="dxf"
    )
    
    return TaskResponse(
        task_id=task.id,
        status="pending",
        result={"nesting_id": nesting.id}
    )


@app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def get_task_status(
    task_id: str,
    request: Request,
    user: User = Depends(get_current_user)
):
    """Get task status."""
    await check_rate_limit(request)
    
    from ..workers.celery_app import celery_app
    
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        return TaskResponse(task_id=task_id, status="pending")
    elif task.state == 'PROGRESS':
        return TaskResponse(
            task_id=task_id,
            status="processing",
            result=task.info
        )
    elif task.state == 'SUCCESS':
        return TaskResponse(
            task_id=task_id,
            status="completed",
            result=task.result
        )
    else:
        return TaskResponse(
            task_id=task_id,
            status="failed",
            error=str(task.info)
        )


@app.get("/api/v1/analyses", response_model=List[AnalysisResponse], tags=["Analysis"])
async def get_analyses(
    request: Request,
    user: User = Depends(lambda: check_permission(Permission.READ_ANALYSIS))
):
    """Get user's analyses."""
    await check_rate_limit(request)
    
    with get_db_session() as session:
        analyses = session.query(Analysis).filter(Analysis.user_id == user.id).all()
        
        return [
            AnalysisResponse(
                id=analysis.id,
                dxf_path=analysis.dxf_path,
                status=analysis.status,
                processing_time=analysis.processing_time,
                total_length=analysis.total_length,
                total_area=analysis.total_area,
                pierce_points=analysis.pierce_points,
                estimated_cost=analysis.estimated_cost,
                created_at=analysis.created_at,
                completed_at=analysis.completed_at
            )
            for analysis in analyses
        ]


@app.get("/api/v1/conversions", response_model=List[ConversionResponse], tags=["Conversion"])
async def get_conversions(
    request: Request,
    user: User = Depends(lambda: check_permission(Permission.READ_CONVERSION))
):
    """Get user's conversions."""
    await check_rate_limit(request)
    
    with get_db_session() as session:
        conversions = session.query(Conversion).filter(Conversion.user_id == user.id).all()
        
        return [
            ConversionResponse(
                id=conversion.id,
                image_path=conversion.image_path,
                dxf_path=conversion.dxf_path,
                preview_path=conversion.preview_path,
                status=conversion.status,
                processing_time=conversion.processing_time,
                polygons_count=conversion.polygons_count,
                contours_count=conversion.contours_count,
                created_at=conversion.created_at,
                completed_at=conversion.completed_at
            )
            for conversion in conversions
        ]


@app.get("/api/v1/nestings", response_model=List[NestingResponse], tags=["Nesting"])
async def get_nestings(
    request: Request,
    user: User = Depends(lambda: check_permission(Permission.READ_NESTING))
):
    """Get user's nestings."""
    await check_rate_limit(request)
    
    with get_db_session() as session:
        nestings = session.query(Nesting).filter(Nesting.user_id == user.id).all()
        
        return [
            NestingResponse(
                id=nesting.id,
                name=nesting.name,
                status=nesting.status,
                processing_time=nesting.processing_time,
                utilization_rate=nesting.utilization_rate,
                parts_count=nesting.parts_count,
                sheets_count=nesting.sheets_count,
                created_at=nesting.created_at,
                completed_at=nesting.completed_at
            )
            for nesting in nestings
        ]


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="WJP ANALYSER API",
        version="1.0.0",
        description="RESTful API for Waterjet Cutting Analysis System",
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    # Add security to all endpoints
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method != "get" or path != "/api/health":
                openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("WJP ANALYSER API starting up")
    
    # Initialize database
    from ..database import init_database
    init_database()
    
    # Validate configuration
    if not config.validate_config():
        logger.warning("Configuration validation failed")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("WJP ANALYSER API shutting down")
    
    # Close database connections
    from ..database import close_database
    close_database()
