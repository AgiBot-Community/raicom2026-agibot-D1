from glob import glob
from pathlib import Path
from setuptools import find_packages, setup


package_name = 'agibot_d1_auto_nav'
speech_model_files = [
    str(path.relative_to(Path(package_name)))
    for path in (Path(package_name) / 'models' / 'speech').rglob('*')
    if path.is_file() and '.cache' not in path.parts
]


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['models/*.npz'] + speech_model_files},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.xml')),
        ('share/' + package_name, ['README.md', 'VISION_VALIDATION.csv']),
        ('share/' + package_name + '/validation', glob('validation/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Project Contributors',
    maintainer_email='maintainers@users.noreply.github.com',
    description='Competition auto navigation bringup for Agibot D1.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'auto_nav_node = agibot_d1_auto_nav.auto_nav_node:main',
        ],
    },
)
