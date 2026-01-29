import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'edie_mic'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('edie_mic/config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'pyaudio'],
    zip_safe=True,
    maintainer='vraptor',
    maintainer_email='khw11044@naver.com',
    description='EDIE Microphone audio streaming node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'final_node = edie_mic.final_main:main',
            'mic_node = edie_mic.mic_main:main',
            'df_node = edie_mic.df_main:main',
            'save_audio = edie_mic.save_audio:main',
            'save_audio_1ch = edie_mic.save_audio_1ch:main',
        ],
    },
)
