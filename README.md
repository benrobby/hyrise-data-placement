 Workload-Driven Data Placement for Capacity-Constrained Multi-Tier In-Memory Database Systems

License: None

## Setup
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


# dm-latency
- sudo umount /mnt/single_ssd
- lsblk
- sudo blockdev --getsize /dev/nvme1n1
- echo "0 7501476528 delay /dev/nvme1n1 0 1" | sudo dmsetup create nvme1n1-latency1ms
- sudo mount /dev/mapper/nvme1n1-latency1ms /mnt/single_ssd


# remove dm-latency
- sudo umount /mnt/single_ssd
- sudo dmsetup ls
- sudo dmsetup remove nvme1n1-latency1ms
- sudo mount /dev/nvme1n1 /mnt/single_ssd

# fio
- sudo fio --name FIO-LATENCY --eta-newline=5s --filename=fio-tempfile.dat --rw=randread --size=2000m --blocksize=4k --iodepth=1 --direct=1 --numjobs=1 --runtime=30 --group_reporting --time_based --thread --refill_buffers
- sudo fio --name FIO-BANDWIDTH --eta-newline=5s --filename=fio-tempfile.dat --rw=read --size=500m --io_size=10g --blocksize=128k --iodepth=64 --direct=1 --numjobs=64 --runtime=30 --group_reporting --time_based --thread --refill_buffers

# add new tiering device
- mount it
- create folder with chmod 777
- add to umap_jemalloc_memory_resource and recompile
- sudo sysctl -w vm.unprivileged_userfaultfd=1 if umap uffd doesn't work
