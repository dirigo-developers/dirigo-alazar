# dirigo-alazar
`dirigo-alazar` provides a AlazarTech–based implementation of the `Digitizer` interface from [Dirigo](https://dirigo.readthedocs.io/). It allows Alazar boards to be used as digitizers within Dirigo acquisition workflows (e.g. galvo–galvo scanning, galvo-resonant scanning, etc.).

> **Note**  
> This is a hardware plugin for Dirigo and is not intended to be used as a standalone library. 

![PyPI](https://img.shields.io/pypi/v/dirigo-alazar)


## Installation
First install the board drivers from the official Alazar website. Then, inside your Python environment (e.g. a conda environment), run:

```bash
pip install dirigo-alazar
```

Verify that your device is recognized in the AlazarDSO utility and functioning before using this plugin.


## Legal Disclaimer
This library is provided "as is" without any warranties, express or implied, including but not limited to the implied warranties of merchantability, fitness for a particular purpose, or non-infringement. The authors are not responsible for any damage to hardware, data loss, or other issues arising from the use or misuse of this library. Users are advised to thoroughly test this library with their specific hardware and configurations before deployment.

This library depends on the AlazarTech API and associated drivers, which must be installed and configured separately. Compatibility and performance depend on the proper installation and operation of these third-party components.

This library is an independent implementation based on publicly available documentation from AlazarTech. It is not affiliated with, endorsed by, or officially supported by AlazarTech.

Use this library at your own risk. Proper operation of hardware and compliance with applicable laws and regulations is the sole responsibility of the user.

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Funding
Development has been supported in part by the National Cancer Institute of the National Institutes of Health under award number R01CA249151.

The content of this repository is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.