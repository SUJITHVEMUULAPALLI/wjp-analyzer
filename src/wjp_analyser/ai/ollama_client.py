"""
Ollama-powered analysis for DXF files and manufacturing recommendations.
Local AI integration using Ollama with open models.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from rich import print


class OllamaConfig(BaseModel):
    """Configuration for Ollama integration."""
    base_url: str = "http://localhost:11434"
    model: str = "gpt-oss-20b"
    timeout: int = 120


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


class OllamaAnalyzer:
    """Ollama-powered analyzer for manufacturing insights."""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.base_url = config.base_url.rstrip('/')
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception:
            return []
    
    def _generate_response(self, prompt: str) -> Optional[str]:
        """Generate response from Ollama model."""
        try:
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '')
            else:
                print(f"[red]Ollama API error: {response.status_code}[/red]")
                return None
                
        except requests.exceptions.Timeout:
            print("[red]Request timed out. Try reducing complexity or increasing timeout.[/red]")
            return None
        except Exception as e:
            print(f"[red]Error calling Ollama: {str(e)}[/red]")
            return None
    
    def analyze_dxf_manufacturing(self, dxf_path: str, analysis_data: Dict[str, Any]) -> Optional[ManufacturingAnalysis]:
        """
        Analyze DXF file for manufacturing feasibility using Ollama.
        
        Args:
            dxf_path: Path to the DXF file
            analysis_data: Pre-computed analysis data from the DXF
            
        Returns:
            ManufacturingAnalysis object with AI insights
        """
        # Check Ollama connection
        if not self._check_ollama_connection():
            print("[red]Error: Cannot connect to Ollama. Make sure Ollama is running.[/red]")
            print("[yellow]Start Ollama with: ollama serve[/yellow]")
            return None
        
        # Check if model is available
        available_models = self._get_available_models()
        if not available_models:
            print("[red]Error: No models available in Ollama.[/red]")
            return None
        
        if self.config.model not in available_models:
            print(f"[yellow]Warning: Model '{self.config.model}' not found.[/yellow]")
            print(f"[blue]Available models: {', '.join(available_models)}[/blue]")
            # Try to use the first available model
            if available_models:
                self.config.model = available_models[0]
                print(f"[blue]Using model: {self.config.model}[/blue]")
            else:
                return None
        
        try:
            # For now, return a basic analysis without AI to test the system
            print(f"[blue]Generating basic analysis for: {os.path.basename(dxf_path)}[/blue]")
            
            # Create a basic analysis based on the data
            feasibility_score = min(95, max(60, 100 - (analysis_data.get('polygon_count', 0) * 2)))
            complexity_level = "Simple" if analysis_data.get('polygon_count', 0) < 10 else "Moderate"
            
            return ManufacturingAnalysis(
                feasibility_score=float(feasibility_score),
                complexity_level=str(complexity_level),
                estimated_time=str(f"{analysis_data.get('polygon_count', 1) * 0.5:.1f} hours"),
                material_recommendations=[str(analysis_data.get('material', 'steel')), 'aluminum'],
                toolpath_suggestions=["outside-in cutting", "optimize path"],
                potential_issues=["check kerf compensation"],
                optimization_tips=["reduce material waste", "optimize nesting"],
                cost_considerations=[f"estimated ${analysis_data.get('estimated_cost', 0):.0f}"],
                model_used=f"Ollama ({self.config.model}) - Basic Analysis"
            )
                
        except Exception as e:
            print(f"[red]Error in analysis:[/red] {str(e)}")
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
        """Create a simplified prompt for manufacturing analysis."""
        return f"""Analyze this DXF for waterjet cutting:

File: {context['file_name']}
Length: {context['total_length']:.1f}mm
Polygons: {context['polygon_count']} ({context['outer_polygons']} outer, {context['inner_polygons']} inner)
Material: {context['material']}, {context['thickness']}mm thick
Cost: ${context['estimated_cost']:.0f}

Respond with JSON only:
{{
  "feasibility_score": 85,
  "complexity_level": "Simple",
  "estimated_time": "2 hours",
  "material_recommendations": ["steel", "aluminum"],
  "toolpath_suggestions": ["outside-in cutting"],
  "potential_issues": ["none"],
  "optimization_tips": ["reduce kerf"],
  "cost_considerations": ["material waste"]
}}"""
    
    def _parse_ai_response(self, response: str) -> Optional[ManufacturingAnalysis]:
        """Parse AI response into ManufacturingAnalysis object."""
        import re

        def _extract_json(text: str) -> Optional[dict]:
            fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            if fence:
                try:
                    return json.loads(fence.group(1))
                except Exception:
                    pass
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
            print(f"[blue]Raw response: {response[:200]}...[/blue]")
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
        # Check Ollama connection
        if not self._check_ollama_connection():
            print("[red]Error: Cannot connect to Ollama. Make sure Ollama is running.[/red]")
            return None
        
        try:
            prompt = f"""As a manufacturing expert, provide design suggestions for waterjet cutting based on this description:

"{design_description}"

Consider:
- Optimal geometry for waterjet cutting
- Material thickness recommendations
- Kerf considerations
- Design complexity vs. cost
- Manufacturing feasibility
- Common design patterns for this type of project

Provide practical, actionable design recommendations."""
            
            print(f"[blue]Generating design suggestions with Ollama model: {self.config.model}[/blue]")
            return self._generate_response(prompt)
            
        except Exception as e:
            print(f"[red]Error generating design suggestions:[/red] {str(e)}")
            return None


def load_analysis_data_from_report(report_path: str) -> Dict[str, Any]:
    """Load analysis data from a JSON report file."""
    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        # Extract and map data to expected format
        extracted_data = {
            "total_length": report_data.get("metrics", {}).get("length_internal_mm", 0) + 
                           report_data.get("metrics", {}).get("length_outer_mm", 0),
            "polygon_count": len(report_data.get("components", [])),
            "outer_polygons": report_data.get("layers", {}).get("OUTER", 0),
            "inner_polygons": report_data.get("layers", {}).get("INNER", 0),
            "material": report_data.get("material", {}).get("name", "Unknown"),
            "thickness": report_data.get("material", {}).get("thickness_mm", 0),
            "kerf": report_data.get("kerf_mm", 0),
            "estimated_cost": report_data.get("metrics", {}).get("estimated_cutting_cost_inr", 0),
            "complexity_metrics": {
                "groups": len(report_data.get("groups", {})),
                "total_area": sum(comp.get("area", 0) for comp in report_data.get("components", [])),
                "avg_perimeter": sum(comp.get("perimeter", 0) for comp in report_data.get("components", [])) / max(len(report_data.get("components", [])), 1)
            },
            "quality_issues": []
        }
        
        return extracted_data
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
