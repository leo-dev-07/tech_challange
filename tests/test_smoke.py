import subprocess
import sys
import os

def test_generate_runs():
    # run the generator with small size
    cmd = [sys.executable, "-m", "app.generate", "--seed", "1", "--companies", "3", "--outdir", "test_output"]
    rc = subprocess.run(cmd, check=False)
    assert rc.returncode == 0
    assert os.path.exists("test_output/companies.json")
