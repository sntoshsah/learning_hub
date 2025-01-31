# Linux System Commands Documentation

## Introduction
Linux provides a powerful command-line interface that allows users to interact with the system efficiently. This documentation covers essential Linux commands and their explanations.

## File and Directory Management

### `ls`
- **Usage**: `ls [options] [directory]`
- **Description**: Lists files and directories in the specified directory.
- **Example**:
  ```sh
  ls -la
  ```
  Lists all files including hidden ones in long format.

### `cd`
- **Usage**: `cd [directory]`
- **Description**: Changes the current working directory.
- **Example**:
  ```sh
  cd /var/www
  ```
  Navigates to the `/var/www` directory.

### `pwd`
- **Usage**: `pwd`
- **Description**: Prints the current working directory.
- **Example**:
  ```sh
  pwd
  ```
  Displays the full path of the current directory.

### `mkdir`
- **Usage**: `mkdir [directory_name]`
- **Description**: Creates a new directory.
- **Example**:
  ```sh
  mkdir new_folder
  ```
  Creates a directory named `new_folder`.

### `rm`
- **Usage**: `rm [options] [file/directory]`
- **Description**: Removes files or directories.
- **Example**:
  ```sh
  rm -r my_folder
  ```
  Recursively removes `my_folder` and its contents.

## User Management

### `whoami`
- **Usage**: `whoami`
- **Description**: Displays the current logged-in user.

### `id`
- **Usage**: `id [username]`
- **Description**: Shows user ID (UID) and group ID (GID).

### `sudo`
- **Usage**: `sudo [command]`
- **Description**: Executes a command as a superuser.

## Process Management

### `ps`
- **Usage**: `ps aux`
- **Description**: Displays running processes.

### `kill`
- **Usage**: `kill [PID]`
- **Description**: Terminates a process by its process ID.

### `htop`
- **Usage**: `htop`
- **Description**: Interactive process viewer.

### `systemctl`
- **Usage**: `systemctl [action] [service]`
- **Description**: Manages system services.
- **Example**:
  ```sh
  systemctl restart apache2
  ```
  Restarts the Apache service.

### `service`
- **Usage**: `service [service_name] [action]`
- **Description**: Controls system services.
- **Example**:
  ```sh
  service nginx status
  ```
  Shows the status of the Nginx service.

## Networking

### `ifconfig`
- **Usage**: `ifconfig`
- **Description**: Displays network interface information.

### `ping`
- **Usage**: `ping [hostname/IP]`
- **Description**: Checks connectivity to a host.

### `wget`
- **Usage**: `wget [URL]`
- **Description**: Downloads files from the internet.

### `ssh`
- **Usage**: `ssh [user]@[host]`
- **Description**: Connects to a remote server via SSH.
- **Example**:
  ```sh
  ssh user@192.168.1.1
  ```
  Connects to the server at `192.168.1.1` as `user`.

### `scp`
- **Usage**: `scp [file] [user]@[host]:[destination]`
- **Description**: Securely copies files between local and remote machines.
- **Example**:
  ```sh
  scp myfile.txt user@192.168.1.1:/home/user/
  ```
  Copies `myfile.txt` to the remote server.

### `rsync`
- **Usage**: `rsync -avz [source] [destination]`
- **Description**: Synchronizes files between local and remote locations efficiently.
- **Example**:
  ```sh
  rsync -avz /local/path/ user@remote:/remote/path/
  ```
  Synchronizes `/local/path/` to `/remote/path/` on the remote server.

## System Monitoring

### `top`
- **Usage**: `top`
- **Description**: Displays real-time system performance.

### `df`
- **Usage**: `df -h`
- **Description**: Shows disk space usage.

### `free`
- **Usage**: `free -m`
- **Description**: Displays memory usage.

### `uptime`
- **Usage**: `uptime`
- **Description**: Shows how long the system has been running.

### `netstat`
- **Usage**: `netstat -tulnp`
- **Description**: Displays active network connections.

## File Permissions

### `chmod`
- **Usage**: `chmod [permissions] [file]`
- **Description**: Changes file permissions.
- **Example**:
  ```sh
  chmod 755 script.sh
  ```
  Grants execute permissions.

### `chown`
- **Usage**: `chown [owner]:[group] [file]`
- **Description**: Changes file owner and group.

## Conclusion
These commands provide a solid foundation for working with Linux systems. Understanding and mastering them will help you efficiently manage files, users, processes, and system resources.
