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



Install ROS2
-------------

To install ROS2, please follow the official installation guide for your platform:

- [ROS2 Installation Guide](https://docs.ros.org/en/jazzy/Installation.html)


Install Gazebo (Harmonic)
-------------

XXX


Install PX4 (Optional)
-----------------------

PX4 is an optional component for SkyRover, primarily for aerial simulations. If you want to integrate PX4 with SkyRover, follow the instructions below:

1. Download PX4 from [PX4 official site](https://px4.io/).
2. Follow the installation steps specific to your system.

Install Navigation2 (Optional)
-------------------------------

If you need advanced path planning and navigation capabilities, you can install Navigation2. To install Navigation2, follow the steps below:

XXX

Train 3D DCC model (or just use pretrained data)
-------------------------------

TODO

.. code-block:: bash

    XXX conda, ray , pytorch...
    XXX
    cd wrapper/dcc_3d/
    python train_dcc_3d.py

Cite Our Work
-------------------------------

XXX
