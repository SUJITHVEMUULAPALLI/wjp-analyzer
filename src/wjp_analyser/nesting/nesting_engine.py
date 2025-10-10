"""
Nesting Optimization Engine

This module provides comprehensive nesting optimization algorithms for maximizing
material utilization in waterjet cutting operations. It includes multiple algorithms
for different optimization strategies and requirements.
"""

from __future__ import annotations

import os
import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely.affinity import translate, rotate
import time

from ..object_management.dxf_object_manager import DXFObject
from ..object_management.layer_manager import CuttingLayer, MaterialSettings, NestingSettings

logger = logging.getLogger(__name__)


class NestingAlgorithm(Enum):
    """Available nesting algorithms."""
    NO_FIT_POLYGON = "nfp"
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    BOTTOM_LEFT_FILL = "bottom_left"
    HYBRID = "hybrid"


class OptimizationStatus(Enum):
    """Status of optimization process."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    CONVERGED = "converged"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class PositionedObject:
    """Represents an object with its position and rotation."""
    object: DXFObject
    x: float
    y: float
    rotation: float
    bounding_box: Tuple[float, float, float, float]
    geometry: Polygon
    is_positioned: bool = True
    positioning_time: float = 0.0


@dataclass
class NestingResult:
    """Result of nesting optimization."""
    success: bool
    algorithm_used: str
    optimization_time: float
    iterations_completed: int
    final_utilization: float
    improvement_percentage: float
    positioned_objects: List[PositionedObject]
    failed_objects: List[DXFObject]
    material_bounds: Tuple[float, float]
    waste_area: float
    waste_percentage: float
    sheets_required: int
    status: OptimizationStatus
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NoFitPolygonAlgorithm:
    """No-Fit Polygon algorithm for precise collision detection."""
    
    def __init__(self, tolerance: float = 0.01):
        """Initialize NFP algorithm."""
        self.tolerance = tolerance
    
    def calculate_nfp(self, polygon_a: Polygon, polygon_b: Polygon) -> List[Polygon]:
        """Calculate No-Fit Polygon for two polygons."""
        try:
            # Get exterior coordinates
            coords_a = list(polygon_a.exterior.coords[:-1])
            coords_b = list(polygon_b.exterior.coords[:-1])
            
            # Calculate Minkowski sum for NFP
            nfp_polygons = []
            
            # For each vertex of polygon A
            for i, vertex_a in enumerate(coords_a):
                # Translate polygon B to vertex A
                translated_b = translate(polygon_b, vertex_a[0], vertex_a[1])
                
                # Calculate difference
                try:
                    diff = polygon_a.difference(translated_b)
                    if isinstance(diff, Polygon) and diff.is_valid:
                        nfp_polygons.append(diff)
                    elif isinstance(diff, MultiPolygon):
                        for poly in diff.geoms:
                            if poly.is_valid:
                                nfp_polygons.append(poly)
                except Exception:
                    continue
            
            return nfp_polygons
            
        except Exception as e:
            logger.warning(f"NFP calculation failed: {e}")
            return []
    
    def is_valid_position(self, polygon: Polygon, positioned_objects: List[PositionedObject], 
                        material_bounds: Tuple[float, float]) -> bool:
        """Check if a position is valid using NFP."""
        try:
            # Check if polygon is within material bounds
            bounds = polygon.bounds
            if (bounds[0] < 0 or bounds[1] < 0 or 
                bounds[2] > material_bounds[0] or bounds[3] > material_bounds[1]):
                return False
            
            # Check collision with positioned objects
            for pos_obj in positioned_objects:
                if polygon.intersects(pos_obj.geometry):
                    return False
            
            return True
            
        except Exception:
            return False


class GeneticAlgorithm:
    """Genetic algorithm for nesting optimization."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, elitism_rate: float = 0.1):
        """Initialize genetic algorithm."""
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.nfp_algorithm = NoFitPolygonAlgorithm()
    
    def create_individual(self, objects: List[DXFObject], 
                        material_bounds: Tuple[float, float]) -> List[PositionedObject]:
        """Create a random individual (solution)."""
        individual = []
        positioned_objects = []
        
        for obj in objects:
            # Random position and rotation
            x = random.uniform(0, material_bounds[0] - obj.geometry.width)
            y = random.uniform(0, material_bounds[1] - obj.geometry.height)
            rotation = random.choice([0, 90, 180, 270])
            
            # Create positioned object
            pos_obj = self._create_positioned_object(obj, x, y, rotation)
            
            # Try to find valid position
            if self._find_valid_position(pos_obj, positioned_objects, material_bounds):
                individual.append(pos_obj)
                positioned_objects.append(pos_obj)
            else:
                # Mark as failed
                pos_obj.is_positioned = False
                individual.append(pos_obj)
        
        return individual
    
    def evaluate_fitness(self, individual: List[PositionedObject], 
                        material_bounds: Tuple[float, float]) -> float:
        """Evaluate fitness of an individual."""
        positioned_objects = [obj for obj in individual if obj.is_positioned]
        
        if not positioned_objects:
            return 0.0
        
        # Calculate material utilization
        total_area = sum(obj.geometry.area for obj in positioned_objects)
        material_area = material_bounds[0] * material_bounds[1]
        utilization = total_area / material_area
        
        # Penalty for failed objects
        failed_penalty = len([obj for obj in individual if not obj.is_positioned]) * 0.1
        
        return max(0, utilization - failed_penalty)
    
    def crossover(self, parent1: List[PositionedObject], 
                  parent2: List[PositionedObject]) -> Tuple[List[PositionedObject], List[PositionedObject]]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, individual: List[PositionedObject], 
               material_bounds: Tuple[float, float]) -> List[PositionedObject]:
        """Mutate an individual."""
        mutated = individual.copy()
        
        for i, pos_obj in enumerate(mutated):
            if random.random() < self.mutation_rate:
                # Mutate position
                if random.random() < 0.5:
                    pos_obj.x = random.uniform(0, material_bounds[0] - pos_obj.geometry.width)
                    pos_obj.y = random.uniform(0, material_bounds[1] - pos_obj.geometry.height)
                
                # Mutate rotation
                if random.random() < 0.3:
                    pos_obj.rotation = random.choice([0, 90, 180, 270])
                
                # Recreate positioned object
                mutated[i] = self._create_positioned_object(
                    pos_obj.object, pos_obj.x, pos_obj.y, pos_obj.rotation
                )
        
        return mutated
    
    def optimize(self, objects: List[DXFObject], 
                material_bounds: Tuple[float, float],
                max_iterations: int = 1000,
                time_limit: float = 300.0) -> NestingResult:
        """Run genetic algorithm optimization."""
        start_time = time.time()
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual(objects, material_bounds)
            population.append(individual)
        
        best_individual = None
        best_fitness = 0.0
        iteration = 0
        
        while iteration < max_iterations and (time.time() - start_time) < time_limit:
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual, material_bounds)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Selection (tournament selection)
            new_population = []
            
            # Elitism
            elite_count = int(self.population_size * self.elitism_rate)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1, material_bounds)
                child2 = self.mutate(child2, material_bounds)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            iteration += 1
        
        # Prepare result
        optimization_time = time.time() - start_time
        positioned_objects = [obj for obj in best_individual if obj.is_positioned]
        failed_objects = [obj.object for obj in best_individual if not obj.is_positioned]
        
        total_area = sum(obj.geometry.area for obj in positioned_objects)
        material_area = material_bounds[0] * material_bounds[1]
        utilization = total_area / material_area if material_area > 0 else 0
        
        return NestingResult(
            success=len(positioned_objects) > 0,
            algorithm_used="genetic",
            optimization_time=optimization_time,
            iterations_completed=iteration,
            final_utilization=utilization,
            improvement_percentage=0.0,  # Will be calculated by caller
            positioned_objects=positioned_objects,
            failed_objects=failed_objects,
            material_bounds=material_bounds,
            waste_area=material_area - total_area,
            waste_percentage=(material_area - total_area) / material_area * 100,
            sheets_required=1,
            status=OptimizationStatus.COMPLETED if iteration < max_iterations else OptimizationStatus.ITERATION_LIMIT
        )
    
    def _create_positioned_object(self, obj: DXFObject, x: float, y: float, rotation: float) -> PositionedObject:
        """Create a positioned object."""
        start_time = time.time()
        
        # Rotate geometry
        rotated_geom = rotate(obj.entity, rotation, origin='centroid')
        
        # Translate to position
        positioned_geom = translate(rotated_geom, x, y)
        
        # Calculate bounding box
        bounds = positioned_geom.bounds
        
        positioning_time = time.time() - start_time
        
        return PositionedObject(
            object=obj,
            x=x,
            y=y,
            rotation=rotation,
            bounding_box=bounds,
            geometry=positioned_geom,
            positioning_time=positioning_time
        )
    
    def _find_valid_position(self, pos_obj: PositionedObject, 
                           positioned_objects: List[PositionedObject],
                           material_bounds: Tuple[float, float]) -> bool:
        """Find a valid position for an object."""
        return self.nfp_algorithm.is_valid_position(
            pos_obj.geometry, positioned_objects, material_bounds
        )
    
    def _tournament_selection(self, population: List[List[PositionedObject]], 
                            fitness_scores: List[float], tournament_size: int = 3) -> List[PositionedObject]:
        """Tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]


class SimulatedAnnealingAlgorithm:
    """Simulated annealing algorithm for nesting optimization."""
    
    def __init__(self, initial_temperature: float = 1000.0, 
                 cooling_rate: float = 0.95, min_temperature: float = 0.1):
        """Initialize simulated annealing."""
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.nfp_algorithm = NoFitPolygonAlgorithm()
    
    def optimize(self, objects: List[DXFObject], 
                material_bounds: Tuple[float, float],
                max_iterations: int = 1000,
                time_limit: float = 300.0) -> NestingResult:
        """Run simulated annealing optimization."""
        start_time = time.time()
        
        # Initialize solution
        current_solution = self._create_initial_solution(objects, material_bounds)
        current_fitness = self._evaluate_solution(current_solution, material_bounds)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        temperature = self.initial_temperature
        iteration = 0
        
        while (iteration < max_iterations and 
               temperature > self.min_temperature and 
               (time.time() - start_time) < time_limit):
            
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(current_solution, material_bounds)
            neighbor_fitness = self._evaluate_solution(neighbor_solution, material_bounds)
            
            # Accept or reject neighbor
            if (neighbor_fitness > current_fitness or 
                random.random() < math.exp((neighbor_fitness - current_fitness) / temperature)):
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                
                if current_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
        
        # Prepare result
        optimization_time = time.time() - start_time
        positioned_objects = [obj for obj in best_solution if obj.is_positioned]
        failed_objects = [obj.object for obj in best_solution if not obj.is_positioned]
        
        total_area = sum(obj.geometry.area for obj in positioned_objects)
        material_area = material_bounds[0] * material_bounds[1]
        utilization = total_area / material_area if material_area > 0 else 0
        
        return NestingResult(
            success=len(positioned_objects) > 0,
            algorithm_used="simulated_annealing",
            optimization_time=optimization_time,
            iterations_completed=iteration,
            final_utilization=utilization,
            improvement_percentage=0.0,
            positioned_objects=positioned_objects,
            failed_objects=failed_objects,
            material_bounds=material_bounds,
            waste_area=material_area - total_area,
            waste_percentage=(material_area - total_area) / material_area * 100,
            sheets_required=1,
            status=OptimizationStatus.COMPLETED if temperature <= self.min_temperature else OptimizationStatus.ITERATION_LIMIT
        )
    
    def _create_initial_solution(self, objects: List[DXFObject], 
                               material_bounds: Tuple[float, float]) -> List[PositionedObject]:
        """Create initial solution."""
        solution = []
        positioned_objects = []
        
        for obj in objects:
            # Try to place object
            pos_obj = self._place_object(obj, positioned_objects, material_bounds)
            solution.append(pos_obj)
            if pos_obj.is_positioned:
                positioned_objects.append(pos_obj)
        
        return solution
    
    def _place_object(self, obj: DXFObject, positioned_objects: List[PositionedObject],
                     material_bounds: Tuple[float, float]) -> PositionedObject:
        """Place an object in the best available position."""
        best_position = None
        best_fitness = -1
        
        # Try different positions
        for _ in range(10):  # Limited attempts
            x = random.uniform(0, material_bounds[0] - obj.geometry.width)
            y = random.uniform(0, material_bounds[1] - obj.geometry.height)
            rotation = random.choice([0, 90, 180, 270])
            
            pos_obj = self._create_positioned_object(obj, x, y, rotation)
            
            if self.nfp_algorithm.is_valid_position(pos_obj.geometry, positioned_objects, material_bounds):
                fitness = self._evaluate_position(pos_obj, positioned_objects, material_bounds)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_position = pos_obj
        
        return best_position if best_position else self._create_positioned_object(obj, 0, 0, 0)
    
    def _evaluate_solution(self, solution: List[PositionedObject], 
                          material_bounds: Tuple[float, float]) -> float:
        """Evaluate a solution."""
        positioned_objects = [obj for obj in solution if obj.is_positioned]
        
        if not positioned_objects:
            return 0.0
        
        total_area = sum(obj.geometry.area for obj in positioned_objects)
        material_area = material_bounds[0] * material_bounds[1]
        utilization = total_area / material_area
        
        return utilization
    
    def _evaluate_position(self, pos_obj: PositionedObject, 
                          positioned_objects: List[PositionedObject],
                          material_bounds: Tuple[float, float]) -> float:
        """Evaluate a position."""
        # Prefer positions closer to origin (bottom-left)
        distance_penalty = math.sqrt(pos_obj.x**2 + pos_obj.y**2) / 1000.0
        
        # Prefer positions that don't waste space
        bounds = pos_obj.geometry.bounds
        width_utilization = bounds[2] - bounds[0]
        height_utilization = bounds[3] - bounds[1]
        space_efficiency = (width_utilization * height_utilization) / (material_bounds[0] * material_bounds[1])
        
        return space_efficiency - distance_penalty
    
    def _generate_neighbor(self, solution: List[PositionedObject], 
                          material_bounds: Tuple[float, float]) -> List[PositionedObject]:
        """Generate a neighbor solution."""
        neighbor = solution.copy()
        
        # Randomly select an object to move
        if not neighbor:
            return neighbor
        
        obj_idx = random.randint(0, len(neighbor) - 1)
        obj = neighbor[obj_idx].object
        
        # Generate new position
        x = random.uniform(0, material_bounds[0] - obj.geometry.width)
        y = random.uniform(0, material_bounds[1] - obj.geometry.height)
        rotation = random.choice([0, 90, 180, 270])
        
        # Create new positioned object
        new_pos_obj = self._create_positioned_object(obj, x, y, rotation)
        
        # Check if position is valid
        other_objects = [pos_obj for i, pos_obj in enumerate(neighbor) if i != obj_idx and pos_obj.is_positioned]
        
        if self.nfp_algorithm.is_valid_position(new_pos_obj.geometry, other_objects, material_bounds):
            neighbor[obj_idx] = new_pos_obj
        
        return neighbor
    
    def _create_positioned_object(self, obj: DXFObject, x: float, y: float, rotation: float) -> PositionedObject:
        """Create a positioned object."""
        start_time = time.time()
        
        # Rotate geometry
        rotated_geom = rotate(obj.entity, rotation, origin='centroid')
        
        # Translate to position
        positioned_geom = translate(rotated_geom, x, y)
        
        # Calculate bounding box
        bounds = positioned_geom.bounds
        
        positioning_time = time.time() - start_time
        
        return PositionedObject(
            object=obj,
            x=x,
            y=y,
            rotation=rotation,
            bounding_box=bounds,
            geometry=positioned_geom,
            positioning_time=positioning_time
        )


class BottomLeftFillAlgorithm:
    """Bottom-left fill algorithm for simple greedy optimization."""
    
    def __init__(self):
        """Initialize bottom-left fill algorithm."""
        self.nfp_algorithm = NoFitPolygonAlgorithm()
    
    def optimize(self, objects: List[DXFObject], 
                material_bounds: Tuple[float, float],
                max_iterations: int = 1000,
                time_limit: float = 300.0) -> NestingResult:
        """Run bottom-left fill optimization."""
        start_time = time.time()
        
        # Sort objects by area (largest first)
        sorted_objects = sorted(objects, key=lambda obj: obj.geometry.area, reverse=True)
        
        positioned_objects = []
        failed_objects = []
        
        for obj in sorted_objects:
            # Try to place object at bottom-left position
            pos_obj = self._place_bottom_left(obj, positioned_objects, material_bounds)
            
            if pos_obj.is_positioned:
                positioned_objects.append(pos_obj)
            else:
                failed_objects.append(obj)
        
        # Prepare result
        optimization_time = time.time() - start_time
        total_area = sum(obj.geometry.area for obj in positioned_objects)
        material_area = material_bounds[0] * material_bounds[1]
        utilization = total_area / material_area if material_area > 0 else 0
        
        return NestingResult(
            success=len(positioned_objects) > 0,
            algorithm_used="bottom_left",
            optimization_time=optimization_time,
            iterations_completed=len(objects),
            final_utilization=utilization,
            improvement_percentage=0.0,
            positioned_objects=positioned_objects,
            failed_objects=failed_objects,
            material_bounds=material_bounds,
            waste_area=material_area - total_area,
            waste_percentage=(material_area - total_area) / material_area * 100,
            sheets_required=1,
            status=OptimizationStatus.COMPLETED
        )
    
    def _place_bottom_left(self, obj: DXFObject, positioned_objects: List[PositionedObject],
                          material_bounds: Tuple[float, float]) -> PositionedObject:
        """Place object at bottom-left position."""
        # Try different rotations
        for rotation in [0, 90, 180, 270]:
            pos_obj = self._create_positioned_object(obj, 0, 0, rotation)
            
            # Find bottom-left position
            for y in np.arange(0, material_bounds[1] - pos_obj.geometry.height, 1.0):
                for x in np.arange(0, material_bounds[0] - pos_obj.geometry.width, 1.0):
                    test_pos_obj = self._create_positioned_object(obj, x, y, rotation)
                    
                    if self.nfp_algorithm.is_valid_position(test_pos_obj.geometry, positioned_objects, material_bounds):
                        return test_pos_obj
        
        # If no valid position found, return failed object
        failed_obj = self._create_positioned_object(obj, 0, 0, 0)
        failed_obj.is_positioned = False
        return failed_obj
    
    def _create_positioned_object(self, obj: DXFObject, x: float, y: float, rotation: float) -> PositionedObject:
        """Create a positioned object."""
        start_time = time.time()
        
        # Rotate geometry
        rotated_geom = rotate(obj.entity, rotation, origin='centroid')
        
        # Translate to position
        positioned_geom = translate(rotated_geom, x, y)
        
        # Calculate bounding box
        bounds = positioned_geom.bounds
        
        positioning_time = time.time() - start_time
        
        return PositionedObject(
            object=obj,
            x=x,
            y=y,
            rotation=rotation,
            bounding_box=bounds,
            geometry=positioned_geom,
            positioning_time=positioning_time
        )


class NestingEngine:
    """Main nesting optimization engine."""
    
    def __init__(self):
        """Initialize nesting engine."""
        self.algorithms = {
            NestingAlgorithm.NO_FIT_POLYGON: NoFitPolygonAlgorithm(),
            NestingAlgorithm.GENETIC: GeneticAlgorithm(),
            NestingAlgorithm.SIMULATED_ANNEALING: SimulatedAnnealingAlgorithm(),
            NestingAlgorithm.BOTTOM_LEFT_FILL: BottomLeftFillAlgorithm()
        }
    
    def optimize_nesting(self, layer: CuttingLayer) -> NestingResult:
        """
        Optimize nesting for a layer.
        
        Args:
            layer: Cutting layer with objects and settings
            
        Returns:
            NestingResult with optimization results
        """
        logger.info(f"Starting nesting optimization for layer '{layer.name}'")
        
        if not layer.objects:
            return NestingResult(
                success=False,
                algorithm_used="none",
                optimization_time=0.0,
                iterations_completed=0,
                final_utilization=0.0,
                improvement_percentage=0.0,
                positioned_objects=[],
                failed_objects=[],
                material_bounds=(layer.material_settings.width, layer.material_settings.height),
                waste_area=layer.material_settings.width * layer.material_settings.height,
                waste_percentage=100.0,
                sheets_required=0,
                status=OptimizationStatus.ERROR,
                errors=["No objects to optimize"]
            )
        
        # Get algorithm
        algorithm_name = layer.nesting_settings.algorithm
        algorithm = self.algorithms.get(NestingAlgorithm(algorithm_name))
        
        if not algorithm:
            return NestingResult(
                success=False,
                algorithm_used=algorithm_name,
                optimization_time=0.0,
                iterations_completed=0,
                final_utilization=0.0,
                improvement_percentage=0.0,
                positioned_objects=[],
                failed_objects=layer.objects,
                material_bounds=(layer.material_settings.width, layer.material_settings.height),
                waste_area=layer.material_settings.width * layer.material_settings.height,
                waste_percentage=100.0,
                sheets_required=0,
                status=OptimizationStatus.ERROR,
                errors=[f"Unknown algorithm: {algorithm_name}"]
            )
        
        # Calculate material bounds
        material_bounds = (layer.material_settings.width, layer.material_settings.height)
        
        # Calculate baseline utilization (original layout)
        baseline_area = sum(obj.geometry.area for obj in layer.objects)
        baseline_utilization = baseline_area / (material_bounds[0] * material_bounds[1])
        
        # Run optimization
        try:
            result = algorithm.optimize(
                objects=layer.objects,
                material_bounds=material_bounds,
                max_iterations=layer.nesting_settings.max_iterations,
                time_limit=layer.nesting_settings.time_limit
            )
            
            # Calculate improvement
            result.improvement_percentage = ((result.final_utilization - baseline_utilization) / 
                                           max(baseline_utilization, 0.001)) * 100
            
            # Update layer status
            layer.status = LayerStatus.OPTIMIZED if result.success else LayerStatus.ERROR
            layer.optimization_result = result
            
            logger.info(f"Nesting optimization completed: {result.final_utilization:.2%} utilization, "
                       f"{result.improvement_percentage:.1f}% improvement")
            
            return result
            
        except Exception as e:
            logger.error(f"Nesting optimization failed: {e}")
            return NestingResult(
                success=False,
                algorithm_used=algorithm_name,
                optimization_time=0.0,
                iterations_completed=0,
                final_utilization=0.0,
                improvement_percentage=0.0,
                positioned_objects=[],
                failed_objects=layer.objects,
                material_bounds=material_bounds,
                waste_area=material_bounds[0] * material_bounds[1],
                waste_percentage=100.0,
                sheets_required=0,
                status=OptimizationStatus.ERROR,
                errors=[f"Optimization failed: {str(e)}"]
            )
    
    def compare_algorithms(self, layer: CuttingLayer) -> Dict[str, NestingResult]:
        """Compare different algorithms for a layer."""
        results = {}
        
        for algorithm_name, algorithm in self.algorithms.items():
            try:
                # Create temporary layer with algorithm settings
                temp_layer = CuttingLayer(
                    layer_id=f"temp_{algorithm_name.value}",
                    name=f"Temp {algorithm_name.value}",
                    layer_type=layer.layer_type,
                    objects=layer.objects,
                    material_settings=layer.material_settings,
                    cutting_settings=layer.cutting_settings,
                    nesting_settings=NestingSettings(algorithm=algorithm_name.value)
                )
                
                result = self.optimize_nesting(temp_layer)
                results[algorithm_name.value] = result
                
            except Exception as e:
                logger.warning(f"Algorithm {algorithm_name.value} failed: {e}")
                results[algorithm_name.value] = NestingResult(
                    success=False,
                    algorithm_used=algorithm_name.value,
                    optimization_time=0.0,
                    iterations_completed=0,
                    final_utilization=0.0,
                    improvement_percentage=0.0,
                    positioned_objects=[],
                    failed_objects=layer.objects,
                    material_bounds=(layer.material_settings.width, layer.material_settings.height),
                    waste_area=layer.material_settings.width * layer.material_settings.height,
                    waste_percentage=100.0,
                    sheets_required=0,
                    status=OptimizationStatus.ERROR,
                    errors=[f"Algorithm failed: {str(e)}"]
                )
        
        return results


# Convenience functions
def create_nesting_engine() -> NestingEngine:
    """Create a new nesting engine."""
    return NestingEngine()


def optimize_layer_nesting(layer: CuttingLayer) -> NestingResult:
    """Convenience function to optimize layer nesting."""
    engine = NestingEngine()
    return engine.optimize_nesting(layer)


def compare_nesting_algorithms(layer: CuttingLayer) -> Dict[str, NestingResult]:
    """Convenience function to compare algorithms."""
    engine = NestingEngine()
    return engine.compare_algorithms(layer)
