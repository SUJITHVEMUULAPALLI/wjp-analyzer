"""
Enhanced Placement Engine for Production-Grade Nesting
=======================================================

Bottom-Left Fill (BLF), NFP refinement, metaheuristics, and spatial indexing
for optimal part placement.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, box
from shapely.affinity import translate, rotate
from shapely.strtree import STRtree
import numpy as np


@dataclass
class PlacementCandidate:
    """Represents a candidate placement for an object."""
    x: float
    y: float
    rotation: float
    bounding_box: Tuple[float, float, float, float]
    score: float  # Higher is better
    utilization: float  # Material utilization if placed here


class BottomLeftFillEngine:
    """Fast Bottom-Left Fill placement algorithm."""
    
    def __init__(
        self,
        sheet_width: float,
        sheet_height: float,
        kerf_margin: float = 1.0,
        allow_rotation: bool = True,
        rotation_steps: List[float] = None,
    ):
        """
        Initialize BLF engine.
        
        Args:
            sheet_width: Sheet width (mm)
            sheet_height: Sheet height (mm)
            kerf_margin: Kerf margin around parts (mm)
            allow_rotation: Allow rotation of parts
            rotation_steps: Rotation angles to try (degrees)
        """
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.kerf_margin = kerf_margin
        self.allow_rotation = allow_rotation
        self.rotation_steps = rotation_steps or [0, 90, 180, 270]
    
    def place_objects(
        self,
        polygons: List[Polygon],
        priority_order: Optional[List[int]] = None,
    ) -> List[PlacementCandidate]:
        """
        Place objects using Bottom-Left Fill strategy.
        
        Args:
            polygons: List of polygons to place
            priority_order: Optional order (indices), otherwise by area (largest first)
            
        Returns:
            List of placement candidates
        """
        if not polygons:
            return []
        
        # Determine order
        if priority_order is None:
            # Sort by area (largest first)
            sorted_indices = sorted(
                range(len(polygons)),
                key=lambda i: abs(polygons[i].area),
                reverse=True,
            )
        else:
            sorted_indices = priority_order
        
        placements: List[PlacementCandidate] = []
        placed_geometries: List[Polygon] = []
        
        # Use STRtree for fast collision detection
        tree = None
        
        for idx in sorted_indices:
            poly = polygons[idx]
            
            # Try different rotations
            best_placement = None
            best_score = -1
            
            for rotation in (self.rotation_steps if self.allow_rotation else [0]):
                # Rotate polygon
                rotated = rotate(poly, rotation, origin=(0, 0))
                bbox = rotated.bounds
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Skip if too large for sheet
                if width > self.sheet_width or height > self.sheet_height:
                    continue
                
                # Find bottom-left position
                placement = self._find_bottom_left_position(
                    rotated,
                    placed_geometries,
                    tree,
                )
                
                if placement:
                    # Calculate score (prefer lower positions, more compact)
                    score = self._calculate_placement_score(placement, self.sheet_height)
                    
                    if score > best_score:
                        best_score = score
                        best_placement = PlacementCandidate(
                            x=placement[0],
                            y=placement[1],
                            rotation=rotation,
                            bounding_box=(
                                placement[0] + bbox[0],
                                placement[1] + bbox[1],
                                placement[0] + bbox[2],
                                placement[1] + bbox[3],
                            ),
                            score=score,
                            utilization=0.0,  # Calculated later
                        )
            
            if best_placement:
                # Place the object
                placed_poly = translate(
                    rotate(poly, best_placement.rotation, origin=(0, 0)),
                    best_placement.x,
                    best_placement.y,
                )
                placed_geometries.append(placed_poly)
                placements.append(best_placement)
                
                # Update STRtree
                tree = STRtree(placed_geometries)
            else:
                # Failed to place
                placements.append(PlacementCandidate(
                    x=0, y=0, rotation=0,
                    bounding_box=(0, 0, 0, 0),
                    score=-1,
                    utilization=0.0,
                ))
        
        return placements
    
    def _find_bottom_left_position(
        self,
        polygon: Polygon,
        existing: List[Polygon],
        tree: Optional[STRtree],
    ) -> Optional[Tuple[float, float]]:
        """Find the bottom-left valid position for a polygon."""
        bbox = polygon.bounds
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Start from bottom-left
        for y in np.arange(0, self.sheet_height - height + 1, self.kerf_margin):
            for x in np.arange(0, self.sheet_width - width + 1, self.kerf_margin):
                # Translate polygon
                test_poly = translate(polygon, x - bbox[0], y - bbox[1])
                
                # Check bounds
                if test_poly.bounds[2] > self.sheet_width or test_poly.bounds[3] > self.sheet_height:
                    continue
                
                # Check collisions using STRtree or direct check
                if tree is not None:
                    # Use STRtree for fast collision detection
                    candidates = tree.query(test_poly)
                    collision = False
                    for existing_poly in candidates:
                        if test_poly.intersects(existing_poly):
                            collision = True
                            break
                    if not collision:
                        return (x - bbox[0], y - bbox[1])
                else:
                    # Direct collision check
                    collision = False
                    for existing_poly in existing:
                        if test_poly.intersects(existing_poly):
                            collision = True
                            break
                    if not collision:
                        return (x - bbox[0], y - bbox[1])
        
        return None
    
    def _calculate_placement_score(
        self,
        position: Tuple[float, float],
        sheet_height: float,
    ) -> float:
        """Calculate placement score (higher is better)."""
        # Prefer lower Y positions (bottom-left)
        # Also consider X position (left is better)
        y_score = (sheet_height - position[1]) / sheet_height
        x_score = 1.0 - (position[0] / (sheet_height * 10))  # Normalize X
        
        return y_score * 0.7 + x_score * 0.3


class NFPRefinementEngine:
    """No-Fit Polygon refinement for precise placement."""
    
    def __init__(self, tolerance: float = 0.01):
        """Initialize NFP refinement engine."""
        self.tolerance = tolerance
    
    def refine_placement(
        self,
        polygon: Polygon,
        placed_polygons: List[Polygon],
        initial_position: Tuple[float, float],
        search_radius: float = 10.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Refine placement using NFP for better fit.
        
        Args:
            polygon: Polygon to place
            placed_polygons: Already placed polygons
            initial_position: Initial (x, y) position
            search_radius: Search radius around initial position (mm)
            
        Returns:
            Refined (x, y) position or None
        """
        # Simple refinement: search around initial position
        # In full implementation, would use NFP for precise placement
        
        best_pos = initial_position
        best_distance = float('inf')
        
        # Search grid around initial position
        step = self.tolerance * 10
        for dx in np.arange(-search_radius, search_radius + step, step):
            for dy in np.arange(-search_radius, search_radius + step, step):
                test_x = initial_position[0] + dx
                test_y = initial_position[1] + dy
                
                test_poly = translate(polygon, test_x, test_y)
                
                # Check collisions
                collision = False
                for placed in placed_polygons:
                    if test_poly.intersects(placed):
                        collision = True
                        break
                
                if not collision:
                    # Prefer positions closer to origin (bottom-left)
                    distance = math.sqrt(test_x**2 + test_y**2)
                    if distance < best_distance:
                        best_distance = distance
                        best_pos = (test_x, test_y)
        
        return best_pos if best_distance < float('inf') else None


class MetaheuristicOptimizer:
    """Metaheuristic optimization for part order and rotation."""
    
    def __init__(
        self,
        algorithm: str = "genetic",
        max_iterations: int = 100,
        population_size: int = 50,
    ):
        """
        Initialize metaheuristic optimizer.
        
        Args:
            algorithm: Algorithm to use ("genetic", "simulated_annealing")
            max_iterations: Maximum iterations
            population_size: Population size (for genetic algorithm)
        """
        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.population_size = population_size
    
    def optimize_order(
        self,
        polygons: List[Polygon],
        sheet_width: float,
        sheet_height: float,
        fitness_func: callable,
    ) -> List[int]:
        """
        Optimize part order for better nesting.
        
        Args:
            polygons: List of polygons
            sheet_width: Sheet width
            sheet_height: Sheet height
            fitness_func: Function to evaluate fitness of an ordering
            
        Returns:
            Optimized order (list of indices)
        """
        if self.algorithm == "genetic":
            return self._genetic_optimize(polygons, sheet_width, sheet_height, fitness_func)
        elif self.algorithm == "simulated_annealing":
            return self._simulated_annealing_optimize(polygons, sheet_width, sheet_height, fitness_func)
        else:
            # Default: return original order
            return list(range(len(polygons)))
    
    def _genetic_optimize(
        self,
        polygons: List[Polygon],
        sheet_width: float,
        sheet_height: float,
        fitness_func: callable,
    ) -> List[int]:
        """Genetic algorithm for order optimization."""
        n = len(polygons)
        if n <= 1:
            return list(range(n))
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            order = list(range(n))
            random.shuffle(order)
            population.append(order)
        
        # Evolve
        for iteration in range(self.max_iterations):
            # Evaluate fitness
            fitness_scores = [
                fitness_func([polygons[i] for i in order])
                for order in population
            ]
            
            # Select best
            sorted_pop = sorted(
                zip(population, fitness_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            
            # Keep top 50%
            top_half = [order for order, _ in sorted_pop[:self.population_size // 2]]
            
            # Crossover and mutation
            new_population = top_half.copy()
            while len(new_population) < self.population_size:
                # Crossover
                parent1, parent2 = random.sample(top_half, 2)
                child = self._crossover_order(parent1, parent2)
                
                # Mutation
                if random.random() < 0.1:
                    child = self._mutate_order(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best
        final_fitness = [
            fitness_func([polygons[i] for i in order])
            for order in population
        ]
        best_idx = max(range(len(population)), key=lambda i: final_fitness[i])
        return population[best_idx]
    
    def _simulated_annealing_optimize(
        self,
        polygons: List[Polygon],
        sheet_width: float,
        sheet_height: float,
        fitness_func: callable,
    ) -> List[int]:
        """Simulated annealing for order optimization."""
        n = len(polygons)
        if n <= 1:
            return list(range(n))
        
        # Start with sorted order (by area)
        current_order = sorted(
            range(n),
            key=lambda i: abs(polygons[i].area),
            reverse=True,
        )
        current_fitness = fitness_func([polygons[i] for i in current_order])
        
        best_order = current_order.copy()
        best_fitness = current_fitness
        
        temperature = 1.0
        cooling_rate = 0.99
        
        for iteration in range(self.max_iterations):
            # Generate neighbor (swap two random elements)
            neighbor_order = current_order.copy()
            i, j = random.sample(range(n), 2)
            neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]
            
            neighbor_fitness = fitness_func([polygons[i] for i in neighbor_order])
            
            # Accept or reject
            if neighbor_fitness > current_fitness or random.random() < math.exp(
                (neighbor_fitness - current_fitness) / temperature
            ):
                current_order = neighbor_order
                current_fitness = neighbor_fitness
                
                if current_fitness > best_fitness:
                    best_order = current_order.copy()
                    best_fitness = current_fitness
            
            # Cool down
            temperature *= cooling_rate
        
        return best_order
    
    def _crossover_order(self, order1: List[int], order2: List[int]) -> List[int]:
        """Crossover two orderings (order crossover)."""
        n = len(order1)
        start, end = sorted(random.sample(range(n), 2))
        
        child = [-1] * n
        child[start:end] = order1[start:end]
        
        # Fill remaining from order2
        idx = 0
        for val in order2:
            if val not in child[start:end]:
                while child[idx] != -1:
                    idx += 1
                child[idx] = val
                idx += 1
        
        return child
    
    def _mutate_order(self, order: List[int]) -> List[int]:
        """Mutate an ordering (swap two elements)."""
        mutated = order.copy()
        i, j = random.sample(range(len(order)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated








