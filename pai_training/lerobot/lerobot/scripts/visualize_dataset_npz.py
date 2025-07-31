import argparse
import numpy as np
import rerun as rr
import time

def visualize_npz(npz_path, mode="local", web_port=9090, ws_port=9876):
    data = np.load(npz_path, allow_pickle=True)
    observations = data["observations"]
    actions = data["actions"]

    assert len(observations) == len(actions), "Mismatched obs/action lengths"

    rr.init("replay_npz", spawn=True)
    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    for t, (obs, act) in enumerate(zip(observations, actions)):
        rr.set_time_sequence("frame", t)

        # pixels (image)
        if "pixels" in obs:
            img = obs["pixels"]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            rr.log("camera/image", rr.Image(img))

        # agent_pos → state
        if "agent_pos" in obs:
            for i, val in enumerate(obs["agent_pos"]):
                rr.log(f"state/agent_pos_{i}", rr.Scalar(val))

        # actions
        for i, val in enumerate(act):
            rr.log(f"action/{i}", rr.Scalar(val))

    print(f"✅ Visualization complete: {npz_path}")
    print("🌐 Open in browser:")
    print(f"http://localhost:{web_port}/?url=rerun%2Bhttp://localhost:{ws_port}/proxy")

    if mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting on Ctrl+C")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--ws-port", type=int, default=9876)
    args = parser.parse_args()

    visualize_npz(args.npz_path, args.mode, args.web_port, args.ws_port)

if __name__ == "__main__":
    main()
