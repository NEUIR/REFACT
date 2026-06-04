import time
from collections import defaultdict
import functools

# Model pricing: (input_price_per_million, output_price_per_million)
MODEL_PRICING = {
    'gemini-2.5-flash': (0.30, 1.20),
    'gemini-2.5-pro': (2.00, 8.00),
    'gemini-3-pro-preview': (4.00, 24.00),
    'deepseek-v3.2': (2.00, 3.00),
    'kimi-k2-instruct': (0.9, 3.6),
    'gemini-3-flash-preview': (1.0, 6.0),
}

class CostLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CostLogger, cls).__new__(cls)
            # Stats per model: {model: {count, input_tokens, output_tokens, cost}}
            cls._instance.model_stats = defaultdict(lambda: {
                'count': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0
            })
            # Stats per function: {function: {count, input_tokens, output_tokens, cost}}
            cls._instance.function_stats = defaultdict(lambda: {
                'count': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0
            })
            cls._instance.total_api_calls = 0
            cls._instance.report_interval = 1024
        return cls._instance

    def _calculate_cost(self, model, input_tokens, output_tokens):
        """Calculate cost based on model pricing"""
        pricing = MODEL_PRICING.get(model, (0.30, 1.20))  # Default to flash pricing
        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]
        return input_cost + output_cost

    def log(self, model, function_name, input_tokens, output_tokens, verbose=False):
        """Log API call tokens and cost"""
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        # Update model stats
        self.model_stats[model]['count'] += 1
        self.model_stats[model]['input_tokens'] += input_tokens
        self.model_stats[model]['output_tokens'] += output_tokens
        self.model_stats[model]['cost'] += cost
        
        # Update function stats
        self.function_stats[function_name]['count'] += 1
        self.function_stats[function_name]['input_tokens'] += input_tokens
        self.function_stats[function_name]['output_tokens'] += output_tokens
        self.function_stats[function_name]['cost'] += cost
        
        # Check report interval
        self.total_api_calls += 1
        if self.total_api_calls % self.report_interval == 0:
            print(f"\n[Auto Report] Reached {self.total_api_calls} API calls")
            self.report()
        
        # Real-time logging
        if verbose:
            print(f"💰 [{model}] {function_name}: in={input_tokens} out={output_tokens} cost=${cost:.6f}")

    def report(self):
        """Print comprehensive cost report"""
        print("\n" + "=" * 100)
        print("=== Cost Report by Model ===")
        print(f"{'Model':<25} | {'Count':<6} | {'Input Tokens':<12} | {'Output Tokens':<12} | {'Avg In':<10} | {'Avg Out':<10} | {'Cost ($)':<10}")
        print("-" * 100)
        
        total_cost = 0.0
        total_input = 0
        total_output = 0
        
        for model, data in sorted(self.model_stats.items(), key=lambda x: x[1]['cost'], reverse=True):
            avg_in = data['input_tokens'] / data['count'] if data['count'] > 0 else 0
            avg_out = data['output_tokens'] / data['count'] if data['count'] > 0 else 0
            print(f"{model:<25} | {data['count']:<6} | {data['input_tokens']:<12} | {data['output_tokens']:<12} | {avg_in:<10.0f} | {avg_out:<10.0f} | {data['cost']:<10.6f}")
            total_cost += data['cost']
            total_input += data['input_tokens']
            total_output += data['output_tokens']
        
        print("-" * 100)
        print(f"{'TOTAL':<25} | {'':<6} | {total_input:<12} | {total_output:<12} | {'':<10} | {'':<10} | {total_cost:<10.6f}")
        
        print("\n=== Cost Report by Function ===")
        print(f"{'Function':<40} | {'Count':<6} | {'Input Tokens':<12} | {'Output Tokens':<12} | {'Avg In':<10} | {'Cost ($)':<10}")
        print("-" * 100)
        
        for func, data in sorted(self.function_stats.items(), key=lambda x: x[1]['cost'], reverse=True):
            avg_in = data['input_tokens'] / data['count'] if data['count'] > 0 else 0
            print(f"{func:<40} | {data['count']:<6} | {data['input_tokens']:<12} | {data['output_tokens']:<12} | {avg_in:<10.0f} | {data['cost']:<10.6f}")
        
        print("=" * 100)
        print(f"\n💵 Total Cost for this run: ${total_cost:.6f}")
        print()

    def reset(self):
        self.model_stats.clear()
        self.function_stats.clear()
        self.total_api_calls = 0
        
    def get_total_cost(self):
        return sum(data['cost'] for data in self.model_stats.values())


# Global instances
cost_logger = CostLogger()
