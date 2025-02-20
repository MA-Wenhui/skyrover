SkyRover
=========

.. image:: logo.png
   :alt: SkyRover Logo

SkyRover, a modular and extensible simulator tailored for cross-domain pathfinding research.

Official Website: https://sites.google.com/view/mapf3d/home

Environment Setup
================

To get started with SkyRover, follow the instructions below to set up your environment.

This project is developed on Ubuntu24.04LTS.


Get the code
-------------

.. code-block:: bash

    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws/src
    git clone https://github.com/MA-Wenhui/skyrover.git



Install ROS2
-------------

To install ROS2, please follow the official installation guide:

- [ROS2 Installation Guide](https://docs.ros.org/en/jazzy/Installation.html)


Install Gazebo (Harmonic)
-------------


Please follow the official installation guide:

- [Gazebo Harmonic Installation](https://gazebosim.org/docs/harmonic/install_ubuntu/)


Install PX4 (Optional)
-----------------------

PX4 is an optional component for SkyRover, primarily for aerial simulations. If you want to integrate PX4 with SkyRover, follow the instructions below:

1. Download PX4 from [PX4 official site](https://px4.io/).
2. Follow the installation steps specific to your system.


Install Navigation2 (Optional)
-------------------------------

Follow the installation steps to install Navigation2 Stack:

- [Navigation2 Doc](https://docs.nav2.org/development_guides/build_docs/index.html#install)



Train 3D DCC model (or just use pretrained data)
-------------------------------

.. code-block:: bash

    conda create -n skyrover python=3.12
    conda actiavte skyrover
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install ray tensorboard 
    cd ~/ros2_ws/src/skyrover
    cd wrapper/dcc_3d/
    python train_dcc_3d.py


Run MAPF 3D
-------------------------------

.. code-block:: bash
    
    cd ~/ros2_ws/
    colcon build --symlink-install
    ros2 run skyrover run_mapf3d --ros-args -p alg:=3dcbs -p pcd:="/path/to/point_cloud/file.pcd"
    ros2 run skyrover run_mapf3d --ros-args -p alg:=3dastar -p pcd:="/path/to/point_cloud/file.pcd"
    ros2 run skyrover run_mapf3d --ros-args -p alg:=3ddcc -p model_path:="/path/to/dcc_model.pth" -p pcd:="/path/to/point_cloud/file.pcd"


Cite Our Work
-------------------------------

.. code-block:: bibtex

    @misc{ma2025skyrovermodularsimulatorcrossdomain,
          title={SkyRover: A Modular Simulator for Cross-Domain Pathfinding}, 
          author={Wenhui Ma and Wenhao Li and Bo Jin and Changhong Lu and Xiangfeng Wang},
          year={2025},
          eprint={2502.08969},
          archivePrefix={arXiv},
          primaryClass={cs.RO},
          url={https://arxiv.org/abs/2502.08969}, 
    }
