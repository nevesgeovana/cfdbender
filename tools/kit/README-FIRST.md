# cfdbender 0.4.0 offline kit
This folder holds an offline Python wheel and one small SU2 check case.
Unzip the whole ZIP into a folder you can write to.
Open your own assistant tool in that folder as you normally do.
Paste the prompt from FIRST-PROMPT.md and let your tool do the setup.
You need Python 3.12 or newer already available on this computer.
You also need your site's prebuilt SU2_CFD executable.
The prompt asks for its path if the computer cannot find it.
The wheel has no runtime dependencies and needs no network to install.
Installation here puts the wheel's Python package into this folder's .venv.
No launcher, alias or PATH change is needed.
The tool plans, runs and collects the supplied smoke matrix locally.
Read SMOKE-CASE.md for the expected iteration_limit outcome.
The HPC profile is a fake template to fill with your own values later.
Keep every setup file and result inside this unzipped folder.
