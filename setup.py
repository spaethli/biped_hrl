"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab>=1.3.0",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
    # scripts/mocap_align.py --c3d and scripts/mocap_marker_template.py read raw Vicon C3D files
    extras_require={"mocap": ["c3d==0.6.0"]},
)
