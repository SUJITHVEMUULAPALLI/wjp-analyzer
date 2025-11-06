"""
Group and Layer Management for WJP Analyser
==========================================

This module provides enhanced organization for groups and layers in the DXF analyzer,
supporting hierarchical groups, smart layer management, and improved selection.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from shapely.geometry import Polygon
import math

@dataclass
class GroupProperties:
    """Properties of a component group."""
    name: str
    count: int
    avg_area: float
    avg_perimeter: float
    avg_vertex_count: float
    complexity: str  # 'simple', 'moderate', 'complex'
    similarity_score: float  # 0-1 score of shape similarity within group
    layer: Optional[str] = None
    subgroups: List[str] = None
    parent_group: Optional[str] = None
    metadata: Dict = None

class GroupManager:
    """Manages component groups and layer organization."""
    
    def __init__(self):
        self.groups: Dict[str, GroupProperties] = {}
        self.layer_assignments: Dict[str, str] = {}  # group -> layer
        self.selection_state: Dict[str, bool] = {}  # group -> selected
        self.hierarchy: Dict[str, List[str]] = {}  # parent -> children
        
    def create_group(self, 
                    name: str,
                    components: List[dict],
                    layer: Optional[str] = None,
                    parent_group: Optional[str] = None) -> GroupProperties:
        """Create a new group from components with automatic property calculation."""
        
        # Calculate average properties
        total_area = sum(c.get('area', 0.0) for c in components)
        total_perimeter = sum(c.get('perimeter', 0.0) for c in components)
        total_vertices = sum(c.get('vertex_count', 0) for c in components)
        count = len(components)
        
        if count == 0:
            return None
            
        avg_area = total_area / count
        avg_perimeter = total_perimeter / count
        avg_vertex_count = total_vertices / count
        
        # Determine complexity
        if avg_vertex_count <= 8:
            complexity = 'simple'
        elif avg_vertex_count <= 50:
            complexity = 'moderate'
        else:
            complexity = 'complex'
            
        # Calculate similarity score between components in group
        similarity_score = self._calculate_similarity_score(components)
        
        group_props = GroupProperties(
            name=name,
            count=count,
            avg_area=avg_area,
            avg_perimeter=avg_perimeter,
            avg_vertex_count=avg_vertex_count,
            complexity=complexity,
            similarity_score=similarity_score,
            layer=layer,
            subgroups=[],
            parent_group=parent_group,
            metadata={}
        )
        
        self.groups[name] = group_props
        self.selection_state[name] = True  # Selected by default
        
        if layer:
            self.layer_assignments[name] = layer
            
        if parent_group:
            if parent_group not in self.hierarchy:
                self.hierarchy[parent_group] = []
            self.hierarchy[parent_group].append(name)
            
        return group_props
    
    def _calculate_similarity_score(self, components: List[dict]) -> float:
        """Calculate a similarity score (0-1) between components in a group."""
        if len(components) <= 1:
            return 1.0
            
        def get_shape_features(comp: dict) -> Tuple[float, float, float]:
            """Extract normalized shape features (area, perimeter, vertex_count)."""
            area = comp.get('area', 0.0)
            perimeter = comp.get('perimeter', 0.0)
            vertex_count = comp.get('vertex_count', 0)
            
            # Normalize to largest values
            max_area = max(c.get('area', 0.0) for c in components)
            max_perimeter = max(c.get('perimeter', 0.0) for c in components)
            max_vertices = max(c.get('vertex_count', 0) for c in components)
            
            return (
                area / max(max_area, 1e-6),
                perimeter / max(max_perimeter, 1e-6),
                vertex_count / max(max_vertices, 1)
            )
        
        # Calculate average distance between all component pairs
        total_dist = 0.0
        count = 0
        
        for i, comp1 in enumerate(components):
            f1 = get_shape_features(comp1)
            for j in range(i + 1, len(components)):
                f2 = get_shape_features(components[j])
                # Euclidean distance between feature vectors
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(f1, f2)))
                total_dist += dist
                count += 1
                
        avg_dist = total_dist / max(count, 1)
        # Convert distance to similarity score (0-1)
        similarity = 1.0 - min(1.0, avg_dist)
        
        return similarity
    
    def create_hierarchical_groups(self, components: List[dict]) -> None:
        """Organize components into a hierarchical group structure."""
        # First level: Group by complexity
        complexity_groups = self._group_by_complexity(components)
        
        # Second level: Group by size within complexity groups
        for complexity, comps in complexity_groups.items():
            parent_name = f"Complexity_{complexity}"
            self.create_group(parent_name, comps)
            
            size_groups = self._group_by_size(comps)
            for size, size_comps in size_groups.items():
                group_name = f"{parent_name}_{size}"
                self.create_group(group_name, size_comps, parent_group=parent_name)
                
                # Third level: Group by similarity within size groups
                similarity_groups = self._group_by_similarity(size_comps)
                for i, sim_comps in enumerate(similarity_groups):
                    sim_name = f"{group_name}_Similarity_{i+1}"
                    self.create_group(sim_name, sim_comps, parent_group=group_name)
    
    def _group_by_complexity(self, components: List[dict]) -> Dict[str, List[dict]]:
        """Group components by vertex count complexity."""
        groups = {'simple': [], 'moderate': [], 'complex': []}
        
        for comp in components:
            vertices = comp.get('vertex_count', 0)
            if vertices <= 8:
                groups['simple'].append(comp)
            elif vertices <= 50:
                groups['moderate'].append(comp)
            else:
                groups['complex'].append(comp)
                
        return groups
    
    def _group_by_size(self, components: List[dict]) -> Dict[str, List[dict]]:
        """Group components by size categories."""
        groups = {'small': [], 'medium': [], 'large': []}
        
        # Find size ranges
        areas = [c.get('area', 0.0) for c in components]
        if not areas:
            return groups
            
        max_area = max(areas)
        
        for comp in components:
            area = comp.get('area', 0.0)
            if area < max_area * 0.1:
                groups['small'].append(comp)
            elif area < max_area * 0.5:
                groups['medium'].append(comp)
            else:
                groups['large'].append(comp)
                
        return groups
    
    def _group_by_similarity(self, components: List[dict], similarity_threshold: float = 0.8) -> List[List[dict]]:
        """Group components by shape similarity."""
        if not components:
            return []
            
        # Initialize groups with first component
        groups = [[components[0]]]
        
        # Try to add each remaining component to an existing group or create new group
        for comp in components[1:]:
            found_group = False
            for group in groups:
                # Calculate similarity with first component in group
                test_components = [group[0], comp]
                similarity = self._calculate_similarity_score(test_components)
                
                if similarity >= similarity_threshold:
                    group.append(comp)
                    found_group = True
                    break
                    
            if not found_group:
                groups.append([comp])
                
        return groups
    
    def suggest_layers(self) -> Dict[str, str]:
        """Suggest appropriate layers for each group based on properties."""
        suggestions = {}
        
        for name, props in self.groups.items():
            if props.layer:  # Skip if already assigned
                continue
                
            # Layer suggestion logic
            if props.complexity == 'simple' and props.avg_vertex_count <= 4:
                suggestions[name] = 'OUTER'
            elif props.complexity == 'simple' and props.similarity_score > 0.9:
                suggestions[name] = 'DECOR'
            elif props.complexity == 'complex':
                suggestions[name] = 'COMPLEX'
            else:
                suggestions[name] = 'INNER'
                
        return suggestions
    
    def set_layer(self, group_name: str, layer: str) -> None:
        """Assign a layer to a group and propagate to subgroups if needed."""
        if group_name not in self.groups:
            return
            
        self.groups[group_name].layer = layer
        self.layer_assignments[group_name] = layer
        
        # Propagate to subgroups if they exist
        if group_name in self.hierarchy:
            for subgroup in self.hierarchy[group_name]:
                self.set_layer(subgroup, layer)
                
    def select_group(self, group_name: str, selected: bool = True) -> None:
        """Select or deselect a group and its subgroups."""
        if group_name not in self.groups:
            return
            
        self.selection_state[group_name] = selected
        
        # Propagate to subgroups
        if group_name in self.hierarchy:
            for subgroup in self.hierarchy[group_name]:
                self.select_group(subgroup, selected)
                
    def get_selected_components(self, components: List[dict]) -> List[dict]:
        """Get all components from selected groups."""
        selected = []
        
        for comp in components:
            group = comp.get('group')
            if group and self.selection_state.get(group, False):
                selected.append(comp)
                
        return selected
    
    def get_layer_stats(self) -> Dict[str, Dict]:
        """Get statistics for each layer."""
        stats = {}
        
        for group_name, layer in self.layer_assignments.items():
            if layer not in stats:
                stats[layer] = {
                    'group_count': 0,
                    'total_components': 0,
                    'avg_complexity': 0.0,
                    'groups': []
                }
                
            group = self.groups[group_name]
            stats[layer]['group_count'] += 1
            stats[layer]['total_components'] += group.count
            stats[layer]['avg_complexity'] += (
                0 if group.complexity == 'simple' else
                1 if group.complexity == 'moderate' else 2
            )
            stats[layer]['groups'].append(group_name)
            
        # Calculate averages
        for layer_stats in stats.values():
            if layer_stats['group_count'] > 0:
                layer_stats['avg_complexity'] /= layer_stats['group_count']
                
        return stats
    
    def export_group_data(self) -> Dict:
        """Export all group data for reporting."""
        return {
            'groups': {name: {
                'name': props.name,
                'count': props.count,
                'avg_area': props.avg_area,
                'avg_perimeter': props.avg_perimeter,
                'avg_vertex_count': props.avg_vertex_count,
                'complexity': props.complexity,
                'similarity_score': props.similarity_score,
                'layer': props.layer,
                'subgroups': props.subgroups,
                'parent_group': props.parent_group,
                'selected': self.selection_state.get(name, False),
                'metadata': props.metadata
            } for name, props in self.groups.items()},
            'hierarchy': self.hierarchy,
            'layer_assignments': self.layer_assignments,
            'layer_stats': self.get_layer_stats()
        }

def create_group_manager(components: List[dict]) -> GroupManager:
    """Factory function to create and initialize a GroupManager with components."""
    manager = GroupManager()
    manager.create_hierarchical_groups(components)
    
    # Apply initial layer suggestions
    suggestions = manager.suggest_layers()
    for group, layer in suggestions.items():
        manager.set_layer(group, layer)
        
    return manager