"""Limit a nominal Cartesian displacement; no measured payload/force claim."""
import math


def next_step(original_step, previous_step, distance, tolerance, *, dt, acceleration, cruise_step):
    values = (original_step, previous_step, distance, tolerance, dt, acceleration, cruise_step)
    if (not all(math.isfinite(float(value)) for value in values)
            or min(original_step, tolerance, dt, acceleration, cruise_step) <= 0.
            or min(previous_step, distance) < 0.):
        raise ValueError('Finite positive timing limits and nonnegative distances are required')
    # Keep a half-tolerance margin so feedback convergence cannot stall at the
    # original strict distance<tolerance stopping boundary.
    remaining = max(0., distance - .5 * tolerance)
    braking_step = math.sqrt(2. * acceleration * remaining) * dt
    return min(original_step, cruise_step,
               previous_step + acceleration * dt * dt, braking_step, distance)
