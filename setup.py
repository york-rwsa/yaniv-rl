import setuptools

setuptools.setup(
    name="yaniv_rl",
    version=0.1,
    packages=setuptools.find_packages(exclude=('tests',)),
    package_data={
        'yaniv_rl': ['game/jsondata/discard_actions.json']
    },
    install_requires=[
        'rlcard>=0.2.5',
        'numpy>=1.16.3',
        'termcolor',
        'tqdm',
        'gym',
        'pettingzoo',
        'ray[rllib]',
        'supersuit'
    ],
    requires_python='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
