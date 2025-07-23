import os

from setuptools import find_packages, setup


# Python 以外のファイルを含める
def package_files(directory, strip_leading):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            package_file = os.path.join(path, filename)
            paths.append(package_file[len(strip_leading):])
    return paths


car_templates = ['templates/*']
web_controller_html = package_files('donkeycar/parts/controllers/templates',
                                    'donkeycar/')

extra_files = car_templates + web_controller_html
print('extra_files', extra_files)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='donkeycar',
      version="4.5.1",
      long_description=long_description,
      description='Self driving library for python.',
      url='https://github.com/autorope/donkeycar',
      author='Will Roscoe, Adam Conway, Tawn Kramer',
      author_email='wroscoe@gmail.com, adam@casaconway.com, tawnkramer@gmail.com',
      license='MIT',
      entry_points={
          'console_scripts': [
              'donkey=donkeycar.management.base:execute_from_command_line',
          ],
      },
      install_requires=[
          'numpy==1.19',
          'pillow',
          'docopt',
          'tornado',
          'requests',
          'h5py',
          'PrettyTable',
          'paho-mqtt',
          "simple_pid",
          'progress',
          'typing_extensions',
          'pyfiglet',
          'psutil',
          "pynmea2",
          'pyserial',
          "utm",
      ],
      extras_require={
          'pi': [
              'picamera',
              'Adafruit_PCA9685',
              'adafruit-circuitpython-lis3dh',
              'adafruit-circuitpython-ssd1306',
              'adafruit-circuitpython-rplidar',
              'RPi.GPIO',
              'imgaug'
          ],
          'nano': [
              'Adafruit_PCA9685',
              'adafruit-circuitpython-lis3dh',
              'adafruit-circuitpython-ssd1306',
              'adafruit-circuitpython-rplidar'
          ],
          'pc': [
              'matplotlib',
              'kivy',
              'protobuf',
              'pandas',
              'pyyaml',
              'plotly',
              'albumentations'
          ],
          'dev': [
              'pytest',
              'pytest-cov',
              'responses',
              'mypy'
          ],
          'ci': ['codecov'],
          'tf': ['tensorflow==2.2.0'],
          'torch': [
              'pytorch',
              'torchvision==0.12',
              'torchaudio',
              'fastai'
          ],
          'mm1': ['pyserial']
      },
      package_data={
          'donkeycar': extra_files,
      },
      include_package_data=True,
      classifiers=[
          # このプロジェクトの成熟度は？一般的な値は次の通り
          #   3 - アルファ
          #   4 - ベータ
          #   5 - プロダクション/安定版
          'Development Status :: 4 - Beta',
          # このプロジェクトの対象者を示す
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          # 希望するライセンスを選択（上記の"license"と一致させること）
          'License :: OSI Approved :: MIT License',
          # サポートする Python バージョンをここで指定。特に次の点を確認
          # Python 2、Python 3、あるいはその両方をサポートするかを明記すること
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      keywords='selfdriving cars donkeycar diyrobocars',
      packages=find_packages(exclude=(['tests', 'docs', 'site', 'env'])),
    )
