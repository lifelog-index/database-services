import platform    # For getting the operating system name
import subprocess  # For executing a shell command

def ping(host, port, timeout=1):
    """
    Returns True if host (str) responds to a ping request.
    Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
    """
    # Option for the number of packets as a function of
    param = '-n' if platform.system().lower()=='windows' else '-c'

    # Building the command. Ex: "ping -c 1 google.com"
    command = ['ping', param, str(timeout), f'{host}:{port}']

    return subprocess.call(command) == 0