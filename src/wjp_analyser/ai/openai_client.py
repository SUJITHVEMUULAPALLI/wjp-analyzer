"""
OpenAI-powered analysis for DXF files and manufacturing recommendations.
"""

import os
import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
from pydantic import BaseModel
from rich import print
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI integration."""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.7


class ManufacturingAnalysis(BaseModel):
    """Results from AI manufacturing analysis."""
    feasibility_score: float
    complexity_level: str
    estimated_time: str
    material_recommendations: List[str]
    toolpath_suggestions: List[str]
    potential_issues: List[str]
    optimization_tips: List[str]
    cost_considerations: List[str]
    model_used: Optional[str] = None


class OpenAIAnalyzer:
    """OpenAI-powered analyzer for manufacturing insights."""
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = None
        
        # Try to load API key from multiple sources
        api_key = self._load_api_key()
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            print("[yellow]Warning: No OpenAI API key found.[/yellow]")
            print("[yellow]Set OPENAI_API_KEY environment variable or use --api-key parameter.[/yellow]")
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from multiple sources in order of preference."""
        # 1. From config object
        if self.config.api_key:
            return self.config.api_key
        
        # 2. From environment variable
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
        
        # 3. From YAML config files under package or repo root
        def _read_key(yaml_path: str) -> Optional[str]:
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    if yaml:
                        data = yaml.safe_load(f) or {}
                        key = (data.get('openai') or {}).get('api_key')
                        if key and key != 'your-api-key-here':
                            return key
                    else:
                        # Minimal fallback parsing
                        txt = f.read()
                        for line in txt.splitlines():
                            if 'api_key:' in line and '"' in line:
                                start = line.find('"') + 1
                                end = line.rfind('"')
                                if start > 0 and end > start:
                                    key = line[start:end]
                                    if key and key != 'your-api-key-here':
                                        return key
            except Exception as e:  # pragma: no cover
                print(f"[yellow]Warning: Could not load {yaml_path}: {e}[/yellow]")
            return None

        # Try package-relative config first
        pkg_config = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config', 'api_keys.yaml'))
        if os.path.exists(pkg_config):
            key = _read_key(pkg_config)
            if key:
                return key

        # Try walking up to find repo root config/config.yaml
        cur = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        for _ in range(5):
            candidate = os.path.join(cur, 'config', 'api_keys.yaml')
            if os.path.exists(candidate):
                key = _read_key(candidate)
                if key:
                    return key
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        
        return None
    
    def analyze_dxf_manufacturing(self, dxf_path: str, analysis_data: Dict[str, Any]) -> Optional[ManufacturingAnalysis]:
        """
        Analyze DXF file for manufacturing feasibility and provide AI recommendations.
        
        Args:
            dxf_path: Path to the DXF file
            analysis_data: Pre-computed analysis data from the DXF
            
        Returns:
            ManufacturingAnalysis object with AI insights
        """
        if not self.client:
            print("[red]Error: OpenAI client not initialized. Check API key.[/red]")
            return None
        
        try:
            # Prepare context for AI analysis
            context = self._prepare_analysis_context(dxf_path, analysis_data)
            
            # Create the prompt for manufacturing analysis
            prompt = self._create_manufacturing_prompt(context)
            
            # Get AI analysis
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert manufacturing engineer specializing in waterjet cutting, CNC machining, and precision manufacturing. Provide detailed, practical analysis and recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse the response
            ai_response = response.choices[0].message.content
            return self._parse_ai_response(ai_response)
            
        except Exception as e:
            print(f"[red]Error in OpenAI analysis:[/red] {str(e)}")
            return None
    
    def _prepare_analysis_context(self, dxf_path: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context data for AI analysis."""
        return {
            "file_name": os.path.basename(dxf_path),
            "total_length": analysis_data.get("total_length", 0),
            "polygon_count": analysis_data.get("polygon_count", 0),
            "outer_polygons": analysis_data.get("outer_polygons", 0),
            "inner_polygons": analysis_data.get("inner_polygons", 0),
            "material": analysis_data.get("material", "Unknown"),
            "thickness": analysis_data.get("thickness", 0),
            "kerf": analysis_data.get("kerf", 0),
            "estimated_cost": analysis_data.get("estimated_cost", 0),
            "complexity_metrics": analysis_data.get("complexity_metrics", {}),
            "quality_issues": analysis_data.get("quality_issues", [])
        }
    
    def _create_manufacturing_prompt(self, context: Dict[str, Any]) -> str:
        """Create a detailed prompt for manufacturing analysis."""
        return f"""
Analyze this DXF file for waterjet cutting manufacturing:

**File Information:**
- File: {context['file_name']}
- Total cutting length: {context['total_length']:.2f} mm
- Polygons: {context['polygon_count']} total ({context['outer_polygons']} outer, {context['inner_polygons']} inner)
- Material: {context['material']}
- Thickness: {context['thickness']} mm
- Kerf: {context['kerf']} mm
- Estimated cost: ${context['estimated_cost']:.2f}

**Quality Issues Found:**
{', '.join(context['quality_issues']) if context['quality_issues'] else 'None detected'}

**Complexity Metrics:**
{json.dumps(context['complexity_metrics'], indent=2)}

Please provide a comprehensive manufacturing analysis including:

1. **Feasibility Score** (0-100): How feasible is this design for waterjet cutting?
2. **Complexity Level**: Simple/Moderate/Complex/Expert
3. **Estimated Time**: Rough estimate for cutting time
4. **Material Recommendations**: Best materials for this design
5. **Toolpath Suggestions**: Optimal cutting strategies
6. **Potential Issues**: Manufacturing challenges to watch for
7. **Optimization Tips**: How to improve the design for better manufacturing
8. **Cost Considerations**: Factors affecting final cost

Format your response as JSON with these exact keys:
- feasibility_score (number 0-100)
- complexity_level (string)
- estimated_time (string)
- material_recommendations (array of strings)
- toolpath_suggestions (array of strings)
- potential_issues (array of strings)
- optimization_tips (array of strings)
- cost_considerations (array of strings)
"""
    
    def _parse_ai_response(self, response: str) -> Optional[ManufacturingAnalysis]:
        """Parse AI response into ManufacturingAnalysis object."""
        def _extract_json(text: str) -> Optional[dict]:
            # Prefer fenced code blocks ```json ... ```
            import re
            fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            if fence:
                try:
                    return json.loads(fence.group(1))
                except Exception:
                    pass
            # Fallback: first to last brace
            si = text.find('{')
            ei = text.rfind('}')
            if si != -1 and ei != -1 and ei > si:
                snippet = text[si:ei+1]
                try:
                    return json.loads(snippet)
                except Exception:
                    pass
            return None

        data = _extract_json(response)
        if not data:
            print("[yellow]Warning: Could not parse AI response as JSON[/yellow]")
            return None

        try:
            return ManufacturingAnalysis(
                feasibility_score=float(data.get("feasibility_score", 0)),
                complexity_level=str(data.get("complexity_level", "Unknown")),
                estimated_time=str(data.get("estimated_time", "Unknown")),
                material_recommendations=list(data.get("material_recommendations", [])),
                toolpath_suggestions=list(data.get("toolpath_suggestions", [])),
                potential_issues=list(data.get("potential_issues", [])),
                optimization_tips=list(data.get("optimization_tips", [])),
                cost_considerations=list(data.get("cost_considerations", [])),
            )
        except Exception as e:
            print(f"[red]Error validating AI response:[/red] {e}")
            return None
    
    def generate_design_suggestions(self, design_description: str) -> Optional[str]:
        """
        Generate design suggestions based on a text description.
        
        Args:
            design_description: Text description of the desired design
            
        Returns:
            AI-generated design suggestions
        """
        if not self.client:
            print("[red]Error: OpenAI client not initialized. Check API key.[/red]")
            return None
        
        try:
            prompt = f"""
As a manufacturing expert, provide design suggestions for waterjet cutting based on this description:

"{design_description}"

Consider:
- Optimal geometry for waterjet cutting
- Material thickness recommendations
- Kerf considerations
- Design complexity vs. cost
- Manufacturing feasibility
- Common design patterns for this type of project

Provide practical, actionable design recommendations.
"""
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert manufacturing engineer and design consultant specializing in waterjet cutting applications."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[red]Error generating design suggestions:[/red] {str(e)}")
            return None


def extract_analysis_data(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize analysis data from an in-memory report dict."""
    try:
        return {
            "total_length": (report_data.get("metrics", {}).get("length_internal_mm", 0)
                              + report_data.get("metrics", {}).get("length_outer_mm", 0)),
            "polygon_count": len(report_data.get("components", [])),
            "outer_polygons": report_data.get("layers", {}).get("OUTER", 0),
            "inner_polygons": report_data.get("layers", {}).get("INNER", 0),
            "material": (report_data.get("material", {}).get("name")
                         if isinstance(report_data.get("material"), dict) else report_data.get("material", "Unknown")),
            "thickness": (report_data.get("material", {}).get("thickness_mm", 0)
                           if isinstance(report_data.get("material"), dict) else 0),
            "kerf": report_data.get("kerf_mm", 0),
            "estimated_cost": report_data.get("metrics", {}).get("estimated_cutting_cost_inr", 0),
            "complexity_metrics": {
                "groups": len(report_data.get("groups", {})),
                "total_area": sum(comp.get("area", 0) for comp in report_data.get("components", [])),
                "avg_perimeter": (sum(comp.get("perimeter", 0) for comp in report_data.get("components", []))
                                   / max(len(report_data.get("components", [])), 1)),
            },
            "quality_issues": report_data.get("quality", {}).get("issues", []),
        }
    except Exception as e:
        print(f"[red]Error extracting analysis data:[/red] {str(e)}")
        return {}


def load_analysis_data_from_report(report_path: str) -> Dict[str, Any]:
    """Load analysis data from a JSON report file path."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        return extract_analysis_data(report_data)
    except Exception as e:
        print(f"[red]Error loading report:[/red] {str(e)}")
        return {}


def save_ai_analysis(analysis: ManufacturingAnalysis, output_path: str):
    """Save AI analysis results to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis.dict(), f, indent=2)
        print(f"[green]AI analysis saved to:[/green] {output_path}")
    except Exception as e:
        print(f"[red]Error saving AI analysis:[/red] {str(e)}")
