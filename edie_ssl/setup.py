import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'edie_ssl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('edie_ssl/config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vraptor',
    maintainer_email='khw11044@naver.com',
    description='Sound Source Localization using GCC-PHAT and SRP-PHAT',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ssl_node = edie_ssl.ssl_main:main',
        ],
    },
)
