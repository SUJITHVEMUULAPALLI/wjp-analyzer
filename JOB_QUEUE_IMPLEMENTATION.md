# Job Queue System Implementation

**Date**: 2025-01-01  
**Status**: ✅ Complete

---

## ✅ Implementation Complete

### 1. Queue Manager (`src/wjp_analyser/api/queue_manager.py`)

**Features**:
- Redis connection management with graceful degradation
- Multiple queue support (default, analysis, conversion, nesting, gcode)
- Job enqueueing with configurable timeouts
- Job status tracking and retrieval
- Worker functions for each job type:
  - `worker_analyze_dxf()` - DXF analysis
  - `worker_convert_image()` - Image conversion
  - `worker_calculate_cost()` - Cost calculation
  - `worker_analyze_csv()` - CSV analysis

**Key Functions**:
- `get_redis_connection()` - Connect to Redis
- `get_queue(queue_name)` - Get queue instance
- `enqueue_job()` - Enqueue background job
- `get_job_status()` - Get job status and result
- `cancel_job()` - Cancel a job
- `is_redis_available()` - Check Redis availability

**Graceful Degradation**:
- If Redis unavailable, functions return None
- API falls back to synchronous processing
- No errors if Redis not installed

---

### 2. Worker Script (`src/wjp_analyser/api/worker.py`)

**Features**:
- Standalone worker script
- Support for multiple queues
- Burst mode (exit when no jobs)
- Named workers
- Queue validation

**Usage**:
```bash
# Start worker for default queue
python -m src.wjp_analyser.api.worker

# Start worker for specific queues
python -m src.wjp_analyser.api.worker --queues analysis,conversion

# Burst mode (exit when no jobs)
python -m src.wjp_analyser.api.worker --burst

# Named worker
python -m src.wjp_analyser.api.worker --name worker-1
```

**Via CLI**:
```bash
wjp worker
wjp worker --queues analysis,conversion
wjp worker --burst
wjp worker --name worker-1
```

---

### 3. FastAPI Integration

**Updated Endpoints**:
- ✅ `/analyze-dxf` - Now supports `async_mode` parameter
- ✅ `/jobs/{job_id}` - Real job status tracking

**Async Flow**:
1. Client calls `/analyze-dxf` with `async_mode=true`
2. API enqueues job and returns `job_id`
3. Worker processes job in background
4. Client polls `/jobs/{job_id}` for status
5. When complete, result available in job status

**Synchronous Fallback**:
- If Redis unavailable, jobs execute synchronously
- No breaking changes for existing clients

---

## 🔧 Setup Requirements

### Install Dependencies
```bash
pip install rq redis
```

### Start Redis Server

**Option 1: Docker (Recommended)**
```bash
docker run -d -p 6379:6379 redis
```

**Option 2: Local Installation**
- Windows: Download from https://github.com/microsoftarchive/redis/releases
- Linux: `sudo apt-get install redis-server`
- macOS: `brew install redis`

**Option 3: Windows Service**
```bash
redis-server --service-install
redis-server --service-start
```

### Verify Redis
```bash
redis-cli ping
# Should return: PONG
```

---

## 🚀 Usage Examples

### 1. Start Worker
```bash
# Terminal 1: Start worker
wjp worker --queues analysis

# Or directly:
python -m src.wjp_analyser.api.worker --queues analysis
```

### 2. Submit Async Job via API
```bash
# Submit async job
curl -X POST http://127.0.0.1:8000/analyze-dxf?async_mode=true \
  -H "Content-Type: application/json" \
  -d '{
    "dxf_path": "path/to/file.dxf",
    "material": "steel",
    "thickness": 6.0
  }'

# Response:
{
  "success": true,
  "job_id": "abc123-def456-..."
}
```

### 3. Check Job Status
```bash
# Check status
curl http://127.0.0.1:8000/jobs/abc123-def456-...

# Response:
{
  "job_id": "abc123-def456-...",
  "status": "running",
  "progress": null,
  "result": null,
  "error": null,
  "created_at": "2025-01-01T12:00:00",
  "completed_at": null
}

# When complete:
{
  "job_id": "abc123-def456-...",
  "status": "completed",
  "progress": null,
  "result": { /* analysis report */ },
  "error": null,
  "created_at": "2025-01-01T12:00:00",
  "completed_at": "2025-01-01T12:05:30"
}
```

---

## 📊 Queue Configuration

### Available Queues
- `default` - General purpose jobs
- `analysis` - DXF analysis jobs
- `conversion` - Image to DXF conversion
- `nesting` - Nesting optimization
- `gcode` - G-code generation

### Queue Selection
Jobs are automatically routed to appropriate queues based on job type.

---

## 🔍 Monitoring

### Check Queue Status
```python
from wjp_analyser.api.queue_manager import get_queue_length

# Check queue lengths
print(f"Analysis queue: {get_queue_length('analysis')} jobs")
print(f"Conversion queue: {get_queue_length('conversion')} jobs")
```

### View Failed Jobs
```python
from wjp_analyser.api.queue_manager import get_failed_jobs

failed = get_failed_jobs('analysis', limit=10)
print(f"Failed jobs: {failed}")
```

### RQ Dashboard (Optional)
Install RQ dashboard for web-based monitoring:
```bash
pip install rq-dashboard
rq-dashboard
# Access at http://localhost:9181
```

---

## 🛠️ Configuration

### Environment Variables
```bash
# Redis connection
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export REDIS_URL=redis://localhost:6379/0
```

### Job Timeouts
- Default: 300 seconds (5 minutes)
- Analysis jobs: 600 seconds (10 minutes)
- Custom: Set in `enqueue_job()` call

### Result TTL
- Default: 3600 seconds (1 hour)
- Jobs results stored for 1 hour after completion
- Custom: Set in `enqueue_job()` call

---

## 📝 Notes

### Graceful Degradation
- System works without Redis (synchronous mode)
- No errors if Redis not available
- Automatic fallback to synchronous processing

### Worker Scaling
- Run multiple workers for parallel processing
- Each worker can listen to multiple queues
- Workers automatically pick up new jobs

### Error Handling
- Failed jobs stored in failed job registry
- Errors included in job status response
- Workers continue processing after failures

---

## 🎯 Next Steps

### Enhancements (Future)
1. **Progress Tracking**: Add progress callbacks for long jobs
2. **WebSocket Updates**: Real-time job status via WebSocket
3. **Priority Queues**: Support job priorities
4. **Scheduled Jobs**: Support delayed/scheduled execution
5. **Job Retries**: Automatic retry for failed jobs

### Integration (Pending)
1. Update Streamlit pages to use async jobs
2. Add job status UI component
3. Implement job cancellation in UI
4. Add job history/audit log

---

## ✅ Status

**Job Queue System**: ✅ **COMPLETE**

- ✅ Queue manager implemented
- ✅ Worker script created
- ✅ FastAPI integration complete
- ✅ CLI integration complete
- ✅ Graceful degradation working
- ✅ Error handling implemented

**Ready for Production**: ✅ Yes (requires Redis)

---

## 🚀 Quick Start

1. **Start Redis**:
   ```bash
   docker run -d -p 6379:6379 redis
   ```

2. **Start Worker**:
   ```bash
   wjp worker --queues analysis,conversion
   ```

3. **Start API**:
   ```bash
   wjp api
   ```

4. **Submit Jobs**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/analyze-dxf?async_mode=true" \
     -H "Content-Type: application/json" \
     -d '{"dxf_path": "file.dxf"}'
   ```

---

**Status**: ✅ Job Queue System Complete  
**Next**: Storage upgrade (S3/MinIO) or continue with Phase 2








