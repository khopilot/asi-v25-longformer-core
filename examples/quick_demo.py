#!/usr/bin/env python3
"""
🚀 ASI V2.5 Quick Demo
⚡ Simple demonstration of 2.44x speedup capability

This script shows the basic usage of ASI V2.5 with minimal setup.
Perfect for testing installation and basic functionality.
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from asi_v25 import create_asi_attention, get_performance_summary, VALIDATED_RESULTS
    from asi_v25_config import ExtremeConfig
except ImportError:
    print("❌ ASI V2.5 not properly installed")
    print("💡 Try: pip install -e .")
    sys.exit(1)

def quick_device_check():
    """Quick device compatibility check"""
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon MPS"
        optimized = "✅ OPTIMIZED"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA: {torch.cuda.get_device_name()}"
        optimized = "⚡ COMPATIBLE"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        optimized = "⚠️ SLOW"
    
    print(f"🔧 Device: {device_name} {optimized}")
    return device

def simple_speedup_demo(device):
    """Simple demonstration of ASI speedup"""
    
    print(f"\n⚡ Quick Speedup Demo")
    print("=" * 30)
    
    # Create ASI attention with validated EXTREME config
    print(f"🚀 Creating ASI attention (EXTREME config)...")
    asi_attention = create_asi_attention(
        dim=768,           # Longformer dimension
        num_heads=12,      # Longformer heads
        use_extreme=True   # Use validated config
    ).to(device)
    
    # Create standard attention for comparison
    print(f"📊 Creating standard attention for comparison...")
    standard_attention = torch.nn.MultiheadAttention(
        embed_dim=768,
        num_heads=12,
        batch_first=True
    ).to(device)
    
    # Test data
    batch_size, seq_len, hidden_size = 2, 1024, 768
    test_data = torch.randn(batch_size, seq_len, hidden_size).to(device)
    
    print(f"🔬 Test data: {test_data.shape}")
    
    # Warmup
    print(f"🔥 Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = standard_attention(test_data, test_data, test_data)
            _ = asi_attention(query=test_data)
    
    # Benchmark standard attention
    print(f"📏 Benchmarking standard attention...")
    times_standard = []
    with torch.no_grad():
        for _ in range(10):
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = standard_attention(test_data, test_data, test_data)
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times_standard.append(end - start)
    
    # Benchmark ASI attention
    print(f"🚀 Benchmarking ASI attention...")
    times_asi = []
    with torch.no_grad():
        for _ in range(10):
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = asi_attention(query=test_data)
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times_asi.append(end - start)
    
    # Calculate results
    avg_standard = sum(times_standard) / len(times_standard) * 1000
    avg_asi = sum(times_asi) / len(times_asi) * 1000
    speedup = avg_standard / avg_asi
    
    throughput_standard = (batch_size * seq_len) / (avg_standard / 1000)
    throughput_asi = (batch_size * seq_len) / (avg_asi / 1000)
    
    # Display results
    print(f"\n📊 QUICK DEMO RESULTS:")
    print(f"  Standard attention: {avg_standard:.1f}ms | {throughput_standard:.0f} tok/s")
    print(f"  ASI attention:      {avg_asi:.1f}ms | {throughput_asi:.0f} tok/s")
    print(f"  🚀 Speedup:         {speedup:.2f}x")
    
    # Assessment
    if speedup >= 2.0:
        status = "🎉 EXCELLENT! Near validated performance!"
    elif speedup >= 1.5:
        status = "✅ GOOD! Solid improvement!"
    elif speedup >= 1.2:
        status = "⚡ MODERATE! Some improvement!"
    else:
        status = "⚠️ LIMITED! Check device compatibility!"
    
    print(f"  🎯 Assessment:      {status}")
    
    return speedup

def display_validated_results():
    """Display the validated EXTREME results"""
    
    print(f"\n🏆 VALIDATED PERFORMANCE (Reference)")
    print("=" * 40)
    
    summary = get_performance_summary()
    print(f"📈 {summary['summary']}")
    
    details = summary['details']
    print(f"🚀 Best speedup: {details['best_speedup']}x")
    print(f"📊 Average speedup: {details['average_speedup']}x")  
    print(f"🔥 Layer coverage: {details['layer_coverage']}%")
    print(f"⚡ Throughput: {details['throughput_tokens_per_sec']:,} tok/s")
    print(f"📏 Max sequence: {details['max_sequence_length']:,} tokens")
    print(f"🖥️ Device: {details['device_optimized']}")
    print(f"🏗️ Architecture: {details['architecture_tested']}")
    
    config = details['configuration']
    print(f"\n🔧 EXTREME Configuration:")
    print(f"  Threshold: {config['asi_threshold']} (ultra-aggressive)")
    print(f"  Feature dim: {config['feature_dim']} (minimal)")
    print(f"  Layers: {config['layers_replaced']}/{config['total_layers']}")

def main():
    """Main demo function"""
    
    print(f"🚀 ASI V2.5 Quick Demo")
    print(f"⚡ Testing basic speedup capability")
    print("=" * 50)
    
    # Device check
    device = quick_device_check()
    
    # Display validated results
    display_validated_results()
    
    # Run simple demo
    try:
        speedup = simple_speedup_demo(device)
        
        print(f"\n🎉 DEMO COMPLETE!")
        print(f"⚡ Achieved {speedup:.2f}x speedup in this quick test")
        print(f"📚 For full validation, run: examples/reproduce_extreme_results.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print(f"💡 Try installing missing dependencies or check device compatibility")

if __name__ == "__main__":
    main() 