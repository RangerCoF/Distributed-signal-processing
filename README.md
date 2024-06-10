# Distributed-signal-processing
Assignment for EE4540 Distributed signal processing <br />

In this project, we designed a randomly distributed sensor network of 100 nodes that can monitor the temperature of a square area of $100 \times 100 \ m^2$. <br />

To calculate the averaging, we implemented the following distributed algorithms: <br />
*  Randomized Gossip
*  Geographic Gossip
*  Gossip with eavesdropping (Greedy Gossip)
*  Broadcast Weighted Gossip
*  Asynchronous ADMM
*  Unicast PDMM
*  Broadcast PDMM

The convergence of these algorithms is shown in `convergence_of_all_averaging_algorithms.mlx`. 

The convergence of Randomized Gossip, Unicast PDMM, and Broadcast PDMM with different transmission failure rates is shown in `gossip_failure.m`, `PDMM_failure_uni.m`, `PDMM_failure_brd.m`, and `failure_plot.m`.

The stability of Randomized Gossip and Broadcast PDMM when a node leaves or joins the sensor network is studied in `avg_edit.m` and `pdmm_edit.m`.
