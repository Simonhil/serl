from setuptools import setup, find_packages

setup(
    name="serl_launcher",
    version="0.1.3",
    description="library for rl experiments",
    url="https://github.com/rail-berkeley/serl",
    author="auth",
    license="MIT",
    install_requires=[
        "zmq",
        "typing",
        "typing_extensions",
        "opencv-python",
        "lz4",
        "agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d",



        "gym>=0.26",
        "pyrealsense2",
        "pymodbus==2.5.3",
        "opencv-python",
        "pyquaternion",
        #"pyspacemouse", 
        "hidapi",
        "pyyaml",
        "rospkg",
        "scipy",
        "requests",
        "flask",
        "defusedxml",
        "pynput",
    ],
    packages=find_packages(),
    zip_safe=False,
)
