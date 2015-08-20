__author__ = 'Pelumi'

exclude = ["BURSAR", "SHOP"]
for name in exclude:
    if name in machineName: # machineName is defined further up in the script.
        inputText.insert("end", machineName + " has been excluded.\n")
    else:
        command = subprocess.Popen( commands here...)


def checkName():
    for name in exclude:
        if machineName in name:
