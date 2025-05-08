"""Class used for development purposes in Simulator"""
import math
import string
import random
from itertools import islice
from copy import deepcopy
from math import log, exp
from geographiclib import geodesic as geolib_gdsc
from agent import Agent

def grad_generator(grad):
    while grad >= -5:
        grad -= 0.2
        if grad <=-5:
            yield -5
        else:
            yield round(grad, 3)

class SpaceBasedParadigm:
    """Mixin to Simulation class for  dev purposes"""

    def prepare_for_simulation(self, agent: Agent):
        self.update_speed_limits(agent)
        self.update_altitude_limits(agent)
        agent.ahead_agent = self.ahead_agent_search(agent, self.by_node_dict()[1])
        if agent.ahead_agent[0]:
            if agent.ahead_agent[1] < self.absolute_minima[1]:
                agent.edge = (agent.edge[0], agent.edge[0])
                agent.position = 0
                agent.speed = 250
                agent.move = 0
                agent.hold_timer = 91
                agent.hold = False
                agent.next_holding_node = None
                if agent.edge[1] in self.agents_in_holding_register:
                    self.agents_in_holding_register[agent.edge[1]].append(agent.id)
                else:
                    self.agents_in_holding_register[agent.edge[1]] = [agent.id]

    def incident_check(self, agent: Agent):
        if agent.in_holding:
            pass
        elif agent.ahead_agent[0]:
            if agent.ahead_agent[0].in_holding:
                pass
            elif agent.ahead_agent[1] < self.absolute_minima[1]:
                if not agent.hold and not agent.ahead_agent[0]:
                    self.inc_count_no_hold += 1
                if not self.incidents:
                    self.incidents = [(agent.id, agent.ahead_agent[0].id, agent.ahead_agent[1])]
                else:
                    self.incidents.append((agent.id, agent.ahead_agent[0].id, agent.ahead_agent[1]))

    def agent_update(self, agent: Agent):
        if agent.in_holding:
            pass
        else:
            self.incident_check(agent)
            if agent.adjusted == 0:
                delta_speed = round((agent.next_speed_limit - agent.speed), 3)
                agent.gradient = round((delta_speed / (agent.distance_to_limit + 0.1)), 12)
            else:
                pass
            if agent.gradient == 0:
                agent.move = self.iteration_duration * agent.speed
            else:
                agent.move = (1 / agent.gradient) * agent.speed * (math.exp(
                        agent.gradient * self.iteration_duration) - 1)

        self.agent_position_update(agent)

        if agent.distance_to_alt_limit < 0:
            self.update_altitude_limits(agent)

        if agent.distance_to_limit < 0:
            self.update_speed_limits(agent)

        self.speed_update(agent)



    def agent_position_update(self, agent: Agent):
        agent.altitude -= (agent.move / (agent.distance_to_alt_limit + 0.1)) * (
                agent.altitude - agent.next_alt_limit
        )  # we try to keep descent constant in space
        # relative to limits position update
        agent.distance_to_alt_limit -= agent.move
        agent.distance_to_limit -= agent.move
        pos = agent.position - agent.move
        # held = self.holding_update_routine(agent, pos)
        if pos >= 0:
            # Part resposible for setting holdings at nodes when reached, the holding magic is in
            # "speed_adjustment_profile_choice"
            if agent.in_holding:
                self.agent_in_holding_update_routine(agent)
            agent.position = pos

        else:
            if agent.on_shortcut:

                if agent.ahead_agent[0]:
                    if agent.ahead_agent[1] < self.absolute_minima[1]:
                        agent.hold = True
                    else:
                        agent.hold = False
                else:
                    agent.hold = False
                agent.shortcut = None

            if agent.edge[1] == self.graph.final_node:
                print(f"{agent.id} cleared for landing")
                self.agents_off_times[agent.id] = self.evolution
                del self.agents_dict[agent.id]
                return
            elif self.shortcut_check(agent, pos):
                pass

            elif self.agent_out_of_holding_routine(agent):
                self.full_history[self.evolution - 1]['hur'] = agent.id
            else:
                agent.edge = (agent.edge[1], self.graph.adjacency[agent.edge[1]][0])
                agent.position = self.graph.adjacency[agent.edge[0]][1] + pos

    def shortcut_check(self, agent, pos):
        if agent.edge[1] in self.shortcuts_dict:
            shortcut_edge = (agent.edge[1], self.shortcuts_dict[agent.edge[1]])
            shortcut_candidate = (shortcut_edge, geolib_gdsc.Geodesic.WGS84.Inverse(
                *self.graph.master_dict[shortcut_edge[0]]['coordinates'],
                *self.graph.master_dict[shortcut_edge[1]]['coordinates'])['s12'] / 1852)

            agent_FAP_w_shortcut = pos + self.FAP_distance(node=shortcut_edge[1]) + shortcut_candidate[1]
            for agt in self.agents_dict.values():
                if agt is agent:
                    continue
                elif abs(agent_FAP_w_shortcut - self.FAP_distance(agent=agt)) < self.absolute_minima[1] + 0.2:
                    return False
            else:
                agent.edge = shortcut_edge
                self.shortcuts_register[shortcut_edge] += 1
                agent.position = shortcut_candidate[1]
                agent.shortcut = shortcut_edge
                return True
        else:
            return False

    def agent_in_holding_update_routine(self, agent: Agent):
        if agent.hold_timer == 1:
            agent.hold_timer = None
            if len(self.agents_in_holding_register[agent.edge[1]]) > 1:
                self.agents_in_holding_register[agent.edge[1]].remove(agent.id)
            else:
                del self.agents_in_holding_register[agent.edge[1]]
            if agent.ahead_agent[1] is not None:
                if agent.ahead_agent[0].in_holding:
                    tt = agent.ahead_agent[0].hold_timer * self.iteration_duration
                    diff = agent.ahead_agent[1] - agent.speed * tt
                    if diff < self.absolute_minima[1] + 0.1:
                        agent.position = 0
                        agent.speed = 220
                        agent.move = 0
                        agent.hold_timer = 90
                        agent.hold = False
                        if agent.edge[1] in self.agents_in_holding_register.keys():
                            self.agents_in_holding_register[agent.edge[1]].append(agent.id)
                        else:
                            self.agents_in_holding_register[agent.edge[1]] = [agent.id]
                elif agent.ahead_agent[1] < self.absolute_minima[1] + 0.1:
                    agent.position = 0
                    agent.speed = 220
                    agent.move = 0
                    agent.hold_timer = 90
                    agent.hold = False
                    if agent.edge[1] in self.agents_in_holding_register.keys():
                        self.agents_in_holding_register[agent.edge[1]].append(agent.id)
                    else:
                        self.agents_in_holding_register[agent.edge[1]] = [agent.id]


        elif agent.hold_timer:
            agent.hold_timer -= 1
            agent.hold = False
            agent.move = 0
            return True

    def agent_out_of_holding_routine(self, agent: Agent):
        if agent.hold:
            if agent.next_holding_node == agent.edge[1] or agent.next_holding_node is None:
                if agent.hold_timer == None:
                    agent.edge = (agent.edge[1], agent.edge[1])
                    agent.position = 0
                    agent.speed = 220
                    agent.move = 0
                    agent.hold_timer = 90
                    agent.hold = False
                    agent.next_holding_node = None

                    if agent.edge[1] in self.agents_in_holding_register.keys():
                        self.agents_in_holding_register[agent.edge[1]].append(agent.id)
                    else:
                        self.agents_in_holding_register[agent.edge[1]] = [agent.id]
                elif agent.hold_timer == 1:
                    agent.hold_timer = None
                    if len(self.agents_in_holding_register[agent.edge[1]]) >= 1:
                        self.agents_in_holding_register[agent.edge[1]].remove(agent.id)
                    else:
                        del self.agents_in_holding_register[agent.edge[1]]

                elif agent.hold_timer:
                    agent.hold_timer -= 1
                    agent.hold = False
                    agent.speed = 220

                    agent.move = 0
                return True
            else:
                return False
        else:
            return False


    def avg_speed(self, current_speed, next_speed_limit, distance_to_limit):
        gradient = (next_speed_limit - current_speed) / distance_to_limit
        tt = log(1 + (gradient * distance_to_limit) / current_speed) / gradient
        return distance_to_limit / tt

    def sapc_single_adjustment_no_hold(self, agt1, agt2, top_speed_agt1=None, minspeed_agt2=220,
                                       minima=5):
        speed = agt1.speed
        if top_speed_agt1 is None:
            top_speed_agt1 = speed
        if agt2.speed < 200:
            agt2.speed = 200
            agt2.gradient = 0

        while speed <= top_speed_agt1:
            t_fixed_speed = agt1.intersection[1] / speed
            if agt2.speed < minspeed_agt2:
                diff_at_intersection = agt2.intersection[1] - t_fixed_speed * agt2.speed
                if diff_at_intersection > minima:
                    agt1.speed = speed
                    agt1.speed_fix = True
                    agt1.gradient = 0
                    agt1.adjusted = 1
                    agt2.gradient = 0
                    agt2.adjusted = -1
                    return True
            else:

                for grad in grad_generator(agt2.gradient):
                    diff_at_intersection = self.difference_at_intersection_fixed_speed(agt2, grad, t_fixed_speed,
                                                                                       min_speed = minspeed_agt2)
                    if diff_at_intersection < minima:
                        continue
                    else:
                        agt2.gradient = grad
                        agt1.speed = speed
                        agt1.speed_fix = True
                        agt1.gradient = 0
                        agt1.adjusted = 1
                        agt2.adjusted = -1
                        return True
            speed += 5
        return False

    def sapc_single_adjustment_both_slow_no_hold(self, agt1, agt2, minspeed_agt2=220, minima=5):
        if agt1.gradient != 0:
            tt = log(1 + (agt1.gradient * agt1.intersection[1]) / agt1.speed) / agt1.gradient
        else:
            tt = agt1.intersection[1]/agt1.speed
        if agt2.speed < 200:
            agt2.speed = 200
            agt2.gradient = 0
        if agt2.speed < minspeed_agt2:
            diff_at_intersection = agt2.intersection[1] - tt * agt2.speed
            if diff_at_intersection > minima:
                agt2.gradient = 0
                agt2.adjusted = -1
                return True
        else:
            for grad in grad_generator(agt2.gradient):
                    diff_at_intersection = self.difference_at_intersection_agt1_adjusted(agt2, grad, tt,
                                                                                    min_speed = minspeed_agt2)
                    if diff_at_intersection < minima:
                        continue
                    else:
                        agt2.gradient = grad
                        agt2.adjusted = -1
                        return True
        return False

    def sapc_holding(self, agt1: Agent, agt2: Agent, dist, top_speed_agt1=None, minspeed_agt2=220):
        if agt2.ahead_agent[0] is agt1:
            if agt1.in_holding:
                for grad in grad_generator(agt2.gradient):
                    diff = self.difference_after_holding_same_leg(agt1, agt2, dist, grad)
                    if diff > self.absolute_minima[1]:
                        if agt1.hold_timer == 1:
                            agt1.adjusted = 0

                        if agt2.hold_timer:
                            pass
                        else:
                            agt2.gradient = grad
                            agt2.adjusted = -1
                        return True

                else:
                    if agt2.speed > minspeed_agt2:
                        agt2.gradient = -5
                        agt2.hold = True
                        agt2.adjusted = -1
                    else:
                        agt2.speed = minspeed_agt2
                        agt2.hold = True
                        agt2.adjusted = -1
                        agt2.gradient = 0

            elif agt2.ahead_agent[1] < self.absolute_minima[1]:
                if agt2.speed > minspeed_agt2:
                    agt2.gradient = -5
                    agt2.adjusted = -1
                    agt2.gradient = 0
                    agt2.hold = True
                else:
                    agt2.speed = minspeed_agt2
                    agt2.adjusted = -1
                    agt2.hold = True
                if agt1.adjusted != -1:
                    agt1.speed = 260 if top_speed_agt1 is None else top_speed_agt1
                    agt1.adjusted = 1
                    agt1.speed_fix = True

            elif agt2.ahead_agent[1] < self.params['intersection_conflict_limit']:
                if agt2.speed > minspeed_agt2:
                    agt2.gradient = -3
                    agt2.adjusted = -1
                else:
                    agt2.speed = minspeed_agt2
                    agt2.gradient = 0
            return True

        elif agt1.in_holding and self.graph.adjacency[agt1.edge[1]][0] == agt2.edge[1]:
            agt1.hold = True
            return True
        elif agt2.in_holding and self.graph.adjacency[agt2.edge[1]][0] == agt1.edge[1]:
            agt2.hold = True
            return True
        return False

    def speed_adjustment_profiles_choice(self, conflicted_pair: tuple):
        agt1, agt2, diff, over_intersection = conflicted_pair
        result = self.sapc_holding(agt1, agt2, diff, top_speed_agt1=260, minspeed_agt2=200)
        if not result:
            # profile choice:
            for minima in [self.params['intersection_conflict_limit'], self.absolute_minima[1]+0.1]:
                if agt1.adjusted == -1:
                    result = self.sapc_single_adjustment_both_slow_no_hold(agt1, agt2, minima=minima)
                else:
                    result = self.sapc_single_adjustment_no_hold(agt1, agt2, top_speed_agt1=260, minima=minima)
                if not result:
                    if agt1.adjusted == -1:
                        result = self.sapc_single_adjustment_both_slow_no_hold(agt1, agt2, minspeed_agt2=200,
                                                                     minima=minima)
                    else:
                        result = self.sapc_single_adjustment_no_hold(agt1, agt2, top_speed_agt1=260, minspeed_agt2=200,
                                                                     minima=minima)
                if result:
                    break
            else:
                if agt1.intersection[0] != agt1.edge[1] and agt2.intersection[0] == agt2.edge[1]:
                    agt1.hold = True
                    agt2.speed = 260
                    agt2.speed_fix = True
                    agt2.adjusted = 1
                    agt1.adjusted = -1
                    result = True
                else:
                    if agt2.speed > 200:
                        agt2.gradient = -5
                    else:
                        agt2.gradient = 0
                        agt2.speed = 200
                    if agt1.adjusted == -1:
                        pass
                    else:
                        agt1.speed = 260
                        agt1.adjusted = 1
                        agt1.gradient = 0
                        agt1.speed_fix = True
                    agt2.adjusted = -1
                    agt2.hold = True
                    result = True

        return result

    def speed_update(self, agent: Agent):
        if agent.speed_fix == True:
            pass
        else:
            agent.speed = agent.speed + agent.gradient * agent.move

    def update_speed_limits(self, agent: Agent):
        (
            agent.acceleration,
            agent.next_speed_limit,
            agent.next_node_w_limit,
            agent.distance_to_limit,
        ) = None, *self.limits_check(agent=agent)

    def adjust_speed(self):
        # used in sim2 single_evolution_procedure
        for pair in self.conflicted_list:
            self.speed_adjustment_profiles_choice(pair)
            for agt in pair[0:2]:
                if agt.adjusted:
                    agt.adjusted_vox = "Fast"

                if agt.adjusted == -1:
                    agt.adjusted_vox = "Slow"


    def difference_at_intersection_agt1_adjusted(self, agt2, grad, tt, min_speed=None):
        if min_speed:
            pass
        else:
            min_speed = agt2.next_speed_limit
        if grad == 0:
            diff_at_intersection = agt2.intersection[1] - tt * agt2.speed
            return diff_at_intersection
        distance_to_achieve_final_speed = (min_speed - agt2.speed) / grad
        if distance_to_achieve_final_speed > agt2.intersection[1]:
            diff_at_intersection = agt2.intersection[1] - (1 / grad) * agt2.speed * \
                                   (exp(grad * (tt)) - 1)
        else:
            tt_in_decc = log(1 + (grad * distance_to_achieve_final_speed) / agt2.speed) / grad
            dist_t_fs_agt2 = (1 / grad) * agt2.speed * (exp(grad * tt_in_decc) - 1) \
                             + min_speed * (tt - tt_in_decc)
            diff_at_intersection = agt2.intersection[1] - dist_t_fs_agt2

        return diff_at_intersection

    def difference_at_intersection_fixed_speed(self, agt2, grad, tt, min_speed=None):
        if min_speed:
            pass
        else:
            min_speed = agt2.next_speed_limit
        if grad == 0:
            diff_at_intersection = agt2.intersection[1] - tt * agt2.speed
            return diff_at_intersection
        distance_to_achieve_final_speed = (min_speed - agt2.speed) / grad
        if distance_to_achieve_final_speed > agt2.intersection[1]:
            diff_at_intersection = agt2.intersection[1] - (1 / grad) * agt2.speed * \
                                   (exp(grad * (tt)) - 1)
        else:
            tt_in_decc = log(1 + (grad * distance_to_achieve_final_speed) / agt2.speed) / grad
            dist_t_fs_agt2 = (1 / grad) * agt2.speed * (exp(grad * tt_in_decc) - 1) \
                             + min_speed * (tt - tt_in_decc)
            diff_at_intersection = agt2.intersection[1] - dist_t_fs_agt2

        return diff_at_intersection

    def difference_after_holding_same_leg(self, agt1: Agent, agt2: Agent, dist, grad=None, speed=None):
        if not speed:
            speed = agt1.speed
        if not grad:
            grad = agt2.gradient
        if agt1.in_holding and not agt2.in_holding:
            tt = agt1.hold_timer * self.iteration_duration
            diff = dist - agt2.speed * (1 / grad) * (exp(grad * (tt)) - 1)
        elif agt2.in_holding and not agt1.in_holding:
            tt = agt2.hold_timer * self.iteration_duration
            if agt1.gradinent:
                diff = dist + agt1.speed * (1 / agt1.gradient) * (exp(agt1.gradient * (tt)) - 1)
            else:
                diff = dist + speed * tt
        elif agt2.in_holding and agt1.in_holding:
            tt = (agt2.hold_timer - agt1.hold_timer) * self.iteration_duration
            if tt >= 0:
                diff = dist + speed * tt
            elif grad == 0:
                diff = dist - tt * agt2.speed
            else:
                diff = dist - agt2.speed * (1 / grad) * (exp(grad * abs(tt)) - 1)
        else:
            diff = False
        return diff

# ========================================================================
# Legacy
# =========================================================================
