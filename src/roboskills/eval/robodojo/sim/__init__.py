"""RoboDojo's container side, run by Isaac Sim's python (`/isaac-sim/python.sh`) in RoboDojo's image (docker/robodojo):
`server.py` (one env per app, and the socket wire), `vector.py` (one scene, one env per layout), `replay.py` (a finished
vector batch from its event log), `state.py` (privileged physical geometry). Only `server`'s wire (`send`/`recv`) runs
on the host, through `client.py`."""
