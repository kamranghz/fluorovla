import torch, time
t=time.time(); torch.cuda.init()
a=torch.randn(1024,1024,device="cuda"); b=(a@a).sum(); torch.cuda.synchronize()
print("warmup ok", torch.cuda.get_device_name(0), float(b), f"{time.time()-t:.1f}s", flush=True)
