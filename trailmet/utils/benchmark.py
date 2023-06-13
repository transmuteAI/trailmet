import time
import torch
from codecarbon import OfflineEmissionsTracker

class BaseBenchmark:
    """
    Class to benchmark a model on a given device.
    Args:
        model (torch.nn.Module): The model to benchmark.
        batch_size (int): The batch size to use for benchmarking.
        input_size (int): The input size to use for benchmarking.
        device_name (str): The device to use for benchmarking.
        warmup_iters (int): The number of warmup iterations to use for benchmarking.
        num_iters (int): The number of iterations to use for benchmarking.
    """
    def __init__(self, model, batch_size, input_size, device_name, warmup_iters=50, num_iters=50):
        self.model = model
        self.batch_size = batch_size
        self.input_size = input_size
        self.device = torch.device(device_name)
        self.warmup_iters = warmup_iters
        self.num_iters = num_iters
        self.model.to(self.device)
    
    def measure_parameters(self):
        """Returns the number of trainable and total parameters in the model."""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return trainable_params, total_params
        
    def measure_size(self):
        """Returns the size of the model in bytes."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        return (param_size + buffer_size)
    
    def measure_throughput(self, batch_size, input_size, warmup_iters=50, num_iters=50):
        """Returns the throughput of the model in images per second."""
        torch.cuda.empty_cache()
        self.model.to(self.device)
        self.model.eval()

        timing = []
        inputs = torch.randn(batch_size, 3, input_size, input_size, device=self.device)

        for _ in range(warmup_iters):
            self.model(inputs)

        torch.cuda.synchronize()
        for _ in range(num_iters):
            start = time.time()
            self.model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)

        timing = torch.as_tensor(timing, dtype=torch.float32)
        
        return (self.batch_size / timing.mean()).item()
    
    def measure_flops(self, input_size):
        """Returns the number of FLOPs in the model."""
        return self.model.get_flops(input_size)
    
    def measure_memory(self, batch_size, input_size, num_iters):
        """Returns the peak memory utilization of the model."""
        self.model.to("cpu")
        torch.cuda.reset_peak_memory_stats(device=self.device)
        pre_mem = torch.cuda.memory_allocated(device=self.device)
        self.model.to(self.device)
        inputs = torch.randn(batch_size, 3, input_size, input_size, device=self.device)
        for i in range(num_iters):
            self.model(inputs)
        max_mem = torch.cuda.max_memory_allocated(device=self.device)
        
        return pre_mem, max_mem
    
    def measure_energy(self, batch_size, input_size, num_iters=50):
        """Returns the energy consumption of the model."""
        self.model.eval()
        inputs = torch.randn(batch_size, 3, input_size, input_size, device=self.device)
        
        with OfflineEmissionsTracker(country_iso_code="IND", log_level="error") as tracker:
            for i in range(num_iters):
                self.model(inputs)
                
        return (tracker._total_energy.kWh, tracker._total_cpu_energy.kWh,
                tracker._total_gpu_energy.kWh, tracker._total_ram_energy.kWh)
            
    def benchmark(self, verbose=True):
        """Prints the benchmark results and returns them as a tuple."""
        _, total_params = self.measure_parameters()
        model_size = self.measure_size()
        throughput = self.measure_throughput(self.batch_size,
                                             self.input_size, 
                                             self.warmup_iters,
                                             self.num_iters)
        
        flops = self.measure_flops(self.input_size)

        pre_mem, max_mem = self.measure_memory(self.batch_size,
                                               self.input_size, 
                                               self.num_iters)
        peak_utilization = max_mem-pre_mem

        energy_consumption, _, _, _ = self.measure_energy(self.batch_size, 
                                                          self.input_size, 
                                                          self.num_iters)
        
        if verbose:
            model_name = self.model.__class__.__name__
            print(f"{model_name}\n"+"-"*(len(model_name)+2))
            print(f"Number of parameters: {total_params}")
            print(f"Model size: {(model_size/1024**2):.2f} MB")
            print(f"Model FLOPs: {flops}")
            print(f"Throughput: {throughput:.2f} img/s")
            print(f"Peak GPU Memory Utilization: {(peak_utilization/1024**2):.2f} MB")
            print(f"Energy Consumption: {energy_consumption:.9f} kWh")

        return (total_params, model_size, throughput, 
                flops, peak_utilization, energy_consumption)

class ModelBenchmark(BaseBenchmark):

    def __init__(self, model, batch_size, input_size, device_name, warmup_iters=50, num_iters=50):
        super().__init__(model, batch_size, input_size, device_name, warmup_iters, num_iters)