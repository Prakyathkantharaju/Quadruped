# Initial setup.

This document will help you get going with the stanford pupper. Will explain common pitfalls in the setup process.


## Setup the connection with the raspberry pi.

- Change your ip address to somthing like this `10.0.0.x` where x is a number between 2 - 9.
- SSH into the raspberry pi.

```bash
ssh pi@10.0.0.20
```

- If the connection is not working. Check the connection using ping.
```bash
ping 10.0.0.20
``` 
If the ping is not working, then the connection is not working. Check the ip address of the raspberry pi by conneciing it to monitor and keyboard.

## Internet and io connection over ssh.

```bash
sudo raspi-config
```
- Add the wifi ssid and password.
- Setup the pigiod over ssh.


## Install all the dependencies.

```bash
sudo ./install_packages.sh
```


## Update the broken dependencies.

```bash
sudo pip3 install --update six
```


# PS4 Connector connection.
- Please follow the official documentation.
