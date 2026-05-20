import sys
import subprocess

def enable_v1():
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    target = """        # ── ALPHA V1 KILL SWITCH (NO-GO VERDICT) ──
        # Alpha V1 is officially disabled to save CPU cycles and prevent negative EV bleeding.
        # Slingger V5 remains 100% active and healthy.
        logger.debug("alpha_v1_disabled_skipping_directional_flow")
        return"""

    replacement = """        # ── ALPHA V1 KILL SWITCH (NO-GO VERDICT) ──
        # Alpha V1 is officially disabled to save CPU cycles and prevent negative EV bleeding.
        # Slingger V5 remains 100% active and healthy.
        # logger.debug("alpha_v1_disabled_skipping_directional_flow")
        # return"""

    if target in content:
        content = content.replace(target, replacement)
        with open('main.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("Successfully commented out V1 kill switch early return in main.py")
        
        # Git commit and push
        subprocess.run(['git', 'add', 'main.py'])
        subprocess.run(['git', 'commit', '-m', 'emergency: re-enable Alpha V1'])
        subprocess.run(['git', 'push', 'origin', 'main'])
        print("Pushed emergency enable to origin/main")
    else:
        print("V1 kill switch early return pattern not found in main.py. Check main.py content.")

if __name__ == "__main__":
    enable_v1()
