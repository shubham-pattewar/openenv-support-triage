import subprocess
with open('groq_out.txt', 'w', encoding='utf-8') as f:
    subprocess.run(['python', 'inference.py'], stdout=f, stderr=subprocess.STDOUT)
