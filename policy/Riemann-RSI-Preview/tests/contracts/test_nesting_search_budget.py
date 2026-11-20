from types import SimpleNamespace
import unittest

from tasks.sort_nesting_dolls_by_size.skills.planning_search_budget.implementation import SearchBudget, IKAttemptCache
from tasks.sort_nesting_dolls_by_size.skills.domain_refusal.implementation import PlanningSearchExhausted


class NestingSearchBudgetTests(unittest.TestCase):
    def test_attempts_are_bounded_before_any_feasible_plan(self):
        budget = SearchBudget(candidates=2)
        budget.consume('candidate'); budget.consume('candidate')
        with self.assertRaises(PlanningSearchExhausted):
            budget.consume('candidate')

    def test_solver_budget_is_shared_across_recursive_fallback(self):
        budget = SearchBudget(solver_calls=2)
        cache = SimpleNamespace(budget=budget)
        for fallback in ('cpu', 'lateral'):
            cache.budget.consume('solver')
        with self.assertRaises(PlanningSearchExhausted):
            cache.budget.consume('solver')

    def test_duplicate_ik_failures_have_two_attempts_but_cpu_fallback_is_fresh(self):
        cache=IKAttemptCache();calls=[]
        key=('right',1,'pose','unchanged-joints',False)
        for _ in range(100):
            self.assertEqual(cache.status(key,lambda:calls.append('native') or 'Fail'),'Fail')
        self.assertEqual(calls,['native','native'])
        cpu=key[:-1]+(True,)
        self.assertEqual(cache.status(cpu,lambda:calls.append('cpu') or 'Success'),'Success')
        self.assertEqual(calls[-1],'cpu')

    def test_ik_success_cache_does_not_cross_start_joints_or_claim_a_path(self):
        cache=IKAttemptCache();key=('left',1,'pose','q0',False)
        self.assertEqual(cache.status(key,lambda:'Success'),'Success')
        self.assertEqual(cache.status(key,lambda:self.fail('Exact successful input repeated')),'Success')
        changed=key[:3]+('q1',False)
        self.assertEqual(cache.status(changed,lambda:'Fail'),'Fail')
        self.assertFalse(cache.stats['infeasibility_proved'])

    def test_planning_stage_budget_does_not_change_official_episode_clock(self):
        clock = [0.]
        budget = SearchBudget(seconds=3., clock=lambda: clock[0])
        clock[0] = 3.
        with self.assertRaises(PlanningSearchExhausted):
            budget.consume('solver')
        self.assertFalse(budget.stats['total_episode_wall_limit'])


if __name__ == '__main__':
    unittest.main()
