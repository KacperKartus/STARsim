from itertools import islice
from copy import deepcopy
from math import log
from geographiclib import geodesic as geolib_gdsc
from agent import Agent


class TimeBasedParadigm:
    """Mixin to Simulation class with alternative Paradigm for simulations."""

    def prepare_for_simulation(self, agent: Agent):
        self.update_speed_limits_and_acceleration(agent)
        self.update_altitude_limits(agent)

    def agent_position_update(self, agent: Agent):
        """Updates position of a given agent, this procedure is run for every agent on a graph in
        :meth:`.Simulation.make_a_move`.

        Args:
            agent(object): Instance of the Agent class
        """
        move = agent.speed * self.iteration_duration

        # we try to keep descent constant in space, however it might change its dynamic as agent approaches
        # the altitude_limit_point and to high agl,
        # Notice that with the below formula descent dynamics are preserved even if agents move past alt_limit
        # in a given evolution
        agent.altitude -= (move / agent.distance_to_alt_limit) * (
                agent.altitude - agent.next_alt_limit
        )
        agent.distance_to_alt_limit -= move
        agent.distance_to_limit -= move
        pos = agent.position - move

        if pos > 0:
            agent.position = pos
        else:
            if agent.holding:
                agent.edge = (agent.edge[1] + "_holding", agent.edge[1])
            elif agent.edge[1] == "FAP/FAF":
                print(f"{agent.id} cleared for landing")
                self.agents_off_times[agent.id] = self.evolution
                del self.agents_dict[agent.id]
                return
            else:
                agent.edge = (agent.edge[1], self.graph.adjacency[agent.edge[1]][0])
                agent.position = self.graph.adjacency[agent.edge[0]][1] + pos

        # Updating of parameters for the next evolution:

        if agent.distance_to_limit <= 0:
            self.update_speed_limits_and_acceleration(agent)

        if agent.distance_to_alt_limit <= 0:
            self.update_altitude_limits(agent)


        # self.update_speed_limits_and_acceleration(agent)
        self.speed_update(agent)


    def speed_update(self, agent: object):
        """Speed update for a given agent, agents' speed can differ in every evolution, whence procedure is run
        every evolution in :meth:`.Simulation.position_update`

        Args:
            agent(object): Agent class' instance
        """
        agent.speed = agent.speed + self.iteration_duration * agent.acceleration

    def avg_speed(self, current_speed, next_speed_limit, distance_to_limit):
        return (1 / 2) * (current_speed + next_speed_limit)


    def get_acceleration(self,
                         agent: Agent,
                         *args,
                         info=False,
                         ):
        """
        Calculates acceleration on a path to the next speed limit
        Args:
            info (bool): When True besides info on the next node_limit will be returned
            agent (Agent): instance of Agent class

        Returns:
            acceleration
        """
        if args:
            current_speed = args
        else:
            current_speed = agent.speed
        (
            next_speed_limit,
            next_node_w_limit,
            distance_to_limit,
        ) = self.limits_check(agent=agent)

        # self.avg_speed(current_speed, next_speed_limit, distance_to_limit, over_space=True)
        # floor division below, guarantees we will "make it" to the right speed before next_node_with_limit
        tt = distance_to_limit / self.avg_speed(current_speed, next_speed_limit, distance_to_limit)
        if info:
            return (next_speed_limit - current_speed) / tt, next_speed_limit, next_node_w_limit, distance_to_limit,
        else:
            return (next_speed_limit - current_speed) / tt

    def update_speed_limits_and_acceleration(self, agent):
        """search for the next speed limit on the Graph"""
        (
            agent.acceleration,
            agent.next_speed_limit,
            agent.next_node_w_limit,
            agent.distance_to_limit,
        ) = self.get_acceleration(agent,
                                  info=True,
                                  )

    def get_position_prediction(self, agent, tt):
        start_speed = agent.speed
        acceleration = agent.acceleration
        reach = start_speed * tt + (1/2) * agent.acceleration * tt ** 2
        node = agent.edge[1]
        distance_to_node = agent.position
        next_node_w_limit = agent.next_node_w_limit
        while reach > distance_to_node:
            if abs(acceleration) > 1:
                tt_to_node = (-start_speed + sqrt(start_speed**2 + 2 * acceleration * distance_to_node))/acceleration
            else:
                tt_to_node = distance_to_node/start_speed
            tt_after_node = tt - tt_to_node
            next_node, distance_node_to_next_node, _ = self.graph.adjacency[node]
            if node == next_node_w_limit:
                start_speed = self.graph.master_dict[node]['speed']
                acceleration, next_speed_limit, next_node_w_limit, distance_to_limit = self.get_acceleration(
                    agent, start_speed, info=True)
                reach = start_speed * tt_after_node + (1 / 2) * acceleration * tt**2
            else:
                reach -= distance_to_node
                distance_to_node = distance_node_to_next_node
            tt = tt_after_node
            previous_node = node
            node = next_node

            if node == 'EOG':
                return None, None, 0

        if reach < distance_to_node:
            position = distance_to_node - reach
            FAP = position + self.graph.master_dict[node]['FAP_distance']
            return (previous_node, node), position, FAP

    def separation_at_intersection(self, agent1: Agent, agent2: Agent):
        if agent1.position <= agent2.position:
            # The default model setup assumes constant deceleration between nodes with speed limits whence below we
            # estimate the acceleration value (to the next node) by ratio of speed and time change, where time change
            # is estimated using the accelataion = const. constraint by using average speed on a given segment.
            agt1 = agent1
            agt2 = agent2
        else:
            agt2 = agent1
            agt1 = agent2

        acc2 = (agt2.speed_at_next_node - agt2.speed) / (
                (agt2.position) / ((agt2.speed + agt2.speed_at_next_node) / 2)
        )
        tt = agt1.position / ((agt1.speed_at_next_node + agt1.speed) / 2)

        return agt2.position - agt2.speed * tt + acc2 * tt ** 2 / 2

    def shortcut_destination_node(self, in_order):
        agent = in_order[0][1]
        if agent.edge[1] in self.graph.shortcut_dict:
            for destination_node in self.graph.shortcut_dict[agent.edge[1]]:
                # intersection, distance_from_intersection = count_back[destination_node]
                # if distance_from_intersection > shortcut_distance + self.minima[1]:
                #     pass
                # else:
                self.clear_of_conflict(agent, destination_node)

                for agent2_id, agent2 in islice(self.agents_dict, idx):
                    if agent2.Fap_distance - FAP_distance < self.absolute_minima[1]:
                        break
                    else:
                        self.taken_shortcut.add(node)
                        return node
        return None

    def clear_of_conflict(self, agent, destination_node):
        """checks if ther is no conflict at the destination node after shortcut"""
        FAP_d = agent.FAP_distance(self.graph)
        shortcut_distance = (
                geolib_gdsc.Geodesic.WGS84.Inverse(
                    *self.locate_agent(agent),
                    *self.graph.master_dict[destination_node]["coordinates"],
                )["s12"]
                * 0.000539957
        )
        tt_at_V_const = shortcut_distance / agent.speed
        for _, agent2 in self.agents_dict.items():
            if agent == agent2:
                continue
            if (
                    shortcut_distance - self.minima[1]
                    < agent2.FAP_distance(self.graph)
                    < FAP_d + self.minima[1]
            ):
                if destination_node in agent2.on_route:
                    (
                        edge_pred,
                        position_pred,
                        FAP_dist_pred,
                    ) = agent2.get_position_prediction(tt_at_V_const, self.graph)

                    FAP_dif = (
                            self.graph.master_dict[destination_node]["FAP_distance"]
                            - FAP_dist_pred
                    )
                    if abs(FAP_dif) > self.minima[1]:
                        continue
                    elif FAP_dif < 0:
                        if (
                                agent2.FAP_distance(self.graph)
                                < self.graph.master_dict[agent.next_intersection][
                            "FAP_distance"
                        ]
                        ):
                            overtake = True
                            return False, overtake, None, None
                    else:
                        overtake = False
                        return False, overtake, FAP_dif, agent2

        return True, None, None, None
