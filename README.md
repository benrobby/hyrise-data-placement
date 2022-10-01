 # Workload-Driven Data Placement for Tierless In-Memory Database Systems

![Screenshot 2022-09-26 at 10 54 56](https://user-images.githubusercontent.com/26392770/192235260-92a8ed54-96ec-46d5-be36-787553141dd1.png)

### Experiment Procedure 

To gather all the relevant for placement decisions, the TieringSelectionRunner first retrieves information about the stored segments in Hyrise. For this, the runner uses the SQL interface (0) and queries the relevant data from the MetaSegmentsTable (1) that contains meta-information about the stored segments. This information includes the segments’ composite identifiers, each consisting of a table id, a column id, and a chunk id. Furthermore, the runner queries the segments’ sizes and access counters. The TieringSelectionRunner communicates with the TieringSelectionPlugin (2) to run a calibration workload (3) and retrieve the results.
To determine the data placement, the TieringSelectionRunner passes the calibration data and segment access data to the appropriate TieringAlgorithm (4) . Our architecture with a shared interface allows us to easily replace and compare the algorithms. We describe in the respective sections the different used optimization solvers that the algorithms use. The user-specified placement selection algorithm then estimates the cost of its decisions using a CostModel (5). The cost models are components independent of the placement selection algorithms. We can easily replace and compare different cost models as all models implement a shared interface.
Once the placement algorithm has determined the data placement for all segments, the assignment is sent via the SQL interface to the TieringSelectionPlugin. The plugin then migrates all segments to their target tiering devices in a single-threaded fashion. In our experiments, we found the runtime of the migration to be negligible and thus did not consider a multi-threaded implementation. However, implementing a parallel segment migration with multiple threads can be a future improvement for larger data sizes. The plugin keeps track of the current location of the segments in a global hashmap to reduce unnecessary copy operations. This implementation could become a point of lock contention on the hashmap for an increasing data size and number of updates to the hashmap. Focusing on placement algorithms and cost models, this procedure of tracking the segments’ locations is sufficient for our considerations. Our measurements did not show any significant runtime overhead of the above-described hashmap implementation.
Once all segments are located on the respective devices according to the data place- ment, the TieringSelectionRunner sends benchmark queries to Hyrise and records the query latencies. The TieringSelectionRunner measures the end-to-end latency, i.e., the wallclock time until it receives the query execution result. For each tiering benchmark execution, we specify the number of clients that concurrently send benchmark queries and the number of cores that Hyrise uses during query processing. Given a set of benchmark queries, we perform one warm-up run that is excluded from the measurements. Per data placement, we run the queries repeatedly for at least twenty minutes or until stable latencies are reached. The reported runtimes are the mean wallclock runtimes of all query executions. The number of executions is the same for all queries.

### Dynamic Workloads
For the dynamic workload experiments, the TieringSelectionRunner asynchronously sends benchmark queries of a dynamic workload to Hyrise (7). At the same time, the AsyncTieringWorker, a component that updates data placements, runs in another thread and periodically sends an updated tiering configuration to the TieringSelectionPlugin in Hyrise (6). This updated data placement is computed according to the standard experimental procedure. However, the AsyncTieringWorker uses windowing to collect access tracking information: The worker bases the placement decisions on only the segment access counter increases that occurred in the last time window. For this, the worker stores the previous segment access counters collected at the end of the previous window. The worker then computes the delta between the current and previous access counters.
In our experiments, we set the window size to two minutes, unless specified otherwise. This choice is subject to the following considerations. The ratio of recent accesses compared to older accesses is what we refer to as the degree of recency. Our solution allows us to tune the degree of recency by increasing or decreasing the window size. The frequency of periodic tiering updates determines the window size.
Choosing the degree of recency is subject to the following general trade-off. On the one hand, if we want to react to workload changes immediately, we need to weigh recent access counters higher so they have a large influence on the placement decisions. On the other hand, we might want our data placement not to overfit a specific short-term workload change. To satisfy the latter requirement, we would need to consider more access counters from further in the past. The desire to not overfit a specific short-term workload stems from the apprehension that the cost of applying a new tiering configuration might exceed the reduction of query processing costs. Our dynamic tiering implementation supports the described windowing-based strategy. Considering other strategies that aggregate access counters is a future work item.


# Setup

## Installation
- git submodule update --init
- install_dependencies in Hyrise
- install umap:
    - git clone git@github.com:LLNL/umap.git
    - mkdir build
    - cd build
    - cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
    - sudo make install -j $(nproc)
- cd tiering_selection_plugin && mkdir cmake-build-release && mkdir cmake-build-debug && cd cmake-build-release
- cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -GNinja -DBUILD_SHARED_LIBS=On
- ninja TieringSelectionPlugin WorkloadHandlerPlugin hyriseServer
- python3.9 -m pip install .
- python3.9 -m tiering_runner -bf ./config/benchmark/compare_lp_greedy_pareto_dram_capacity_tpch.json -rf ./config/meta/run_benchmarks_only.json


## dm-latency to increase device latency
- sudo umount /mnt/single_ssd
- lsblk
- sudo blockdev --getsize /dev/nvme1n1
- echo "0 7501476528 delay /dev/nvme1n1 0 1" | sudo dmsetup create nvme1n1-latency1ms
- sudo mount /dev/mapper/nvme1n1-latency1ms /mnt/single_ssd

## remove dm-latency
- sudo umount /mnt/single_ssd
- sudo dmsetup ls
- sudo dmsetup remove nvme1n1-latency1ms
- sudo mount /dev/nvme1n1 /mnt/single_ssd

## fio measurements
This measures the device's bandwidth and latency.
- fio --name FIO-LATENCY --filename=f --rw=randread --size=1000m --blocksize=4k --iodepth=1 --direct=1 --numjobs=1 --runtime=60 --group_reporting --time_based --thread --refill_buffers
- fio --name FIO-BANDWIDTH --filename=f --rw=read --size=1000m --blocksize=128k --iodepth=64 --direct=1 --numjobs=64 --runtime=60 --group_reporting --time_based --thread --refill_buffers

## add new tiering device
- mount it
- create folder with chmod 777
- add to umap_jemalloc_memory_resource and recompile
- sudo sysctl -w vm.unprivileged_userfaultfd=1 if umap uffd doesn't work

License: None

tiering_selection_plugin/workload_handler.(cpp|hpp) from https://github.com/Bouncner/encoding_selection_plugin/blob/main/workload_handler.cpp
