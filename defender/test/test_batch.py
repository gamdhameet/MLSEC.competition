"""
Batch test malware detector with multiple files
"""
import requests
import os
import json
import time
from pathlib import Path

def test_batch_files(folder_path, num_files=50, output_file="results.json"):
    """
    Test multiple files against the malware detector API
    
    Args:
        folder_path: Path to folder containing files named 1, 2, 3, etc.
        num_files: Number of files to test (default 50)
        output_file: Where to save results
    """
    
    # Check if API is healthy
    # try:
    #     response = requests.get('http://localhost:8080/health', timeout=5)
    #     print(f"✅ API Health Check: {response.json()}")
    #     print()
    # except Exception as e:
    #     print(f"❌ API not reachable: {e}")
    #     print("Make sure Docker container is running!")
    #     return
    
    results = []
    errors = []
    
    print(f"Testing {num_files} files from {folder_path}")
    print("=" * 70)
    
    for i in range(1, num_files + 1):
        file_path = os.path.join(folder_path, str(i))
        
        # Check if file exists (try with and without .exe extension)
        if not os.path.exists(file_path):
            file_path = os.path.join(folder_path, f"{i}.exe")
        
        if not os.path.exists(file_path):
            print(f"⚠️  File {i} not found, skipping...")
            errors.append({"file": i, "error": "File not found"})
            continue
        
        try:
            # Send file to API
            with open(file_path, 'rb') as f:
                files = {'file': (str(i), f, 'application/octet-stream')}
                response = requests.post(
                    'http://localhost:8080/predict', 
                    files=files, 
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'file_id': i,
                    'file_path': file_path,
                    'prediction': result['prediction'],
                    'label': result['label'],
                    'probability': result['probability'],
                    'threshold': result['threshold']
                })
                
                # Print result
                symbol = "🦠" if result['label'] == 'malware' else "✅"
                print(f"{symbol} File {i:3d}: {result['label'].upper():8s} "
                      f"(prob={result['probability']:.4f})")
                
            else:
                print(f"❌ File {i}: API Error {response.status_code}")
                errors.append({
                    "file": i, 
                    "error": f"HTTP {response.status_code}: {response.text}"
                })
        
        except Exception as e:
            print(f"❌ File {i}: {str(e)}")
            errors.append({"file": i, "error": str(e)})
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    print("\n" + "=" * 70)
    
    # Summary statistics
    total_tested = len(results)
    malware_count = sum(1 for r in results if r['prediction'] == 1)
    benign_count = sum(1 for r in results if r['prediction'] == 0)
    
    print(f"\n📊 SUMMARY")
    print(f"Total files tested: {total_tested}")
    print(f"Malware detected: {malware_count} ({malware_count/total_tested*100:.1f}%)")
    print(f"Benign detected: {benign_count} ({benign_count/total_tested*100:.1f}%)")
    print(f"Errors: {len(errors)}")
    
    # Save results to JSON
    output_data = {
        'summary': {
            'total_tested': total_tested,
            'malware_count': malware_count,
            'benign_count': benign_count,
            'error_count': len(errors)
        },
        'results': results,
        'errors': errors
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Show distribution of probabilities
    if results:
        print(f"\n📈 PROBABILITY DISTRIBUTION")
        malware_probs = [r['probability'] for r in results if r['prediction'] == 1]
        benign_probs = [r['probability'] for r in results if r['prediction'] == 0]
        
        if malware_probs:
            print(f"Malware - Min: {min(malware_probs):.4f}, "
                  f"Max: {max(malware_probs):.4f}, "
                  f"Avg: {sum(malware_probs)/len(malware_probs):.4f}")
        
        if benign_probs:
            print(f"Benign  - Min: {min(benign_probs):.4f}, "
                  f"Max: {max(benign_probs):.4f}, "
                  f"Avg: {sum(benign_probs)/len(benign_probs):.4f}")
    
    return results, errors


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test malware detector')
    parser.add_argument('folder', help='Folder containing test files')
    parser.add_argument('--num-files', type=int, default=50,
                       help='Number of files to test (default: 50)')
    parser.add_argument('--output', default='results.json',
                       help='Output file for results (default: results.json)')
    
    args = parser.parse_args()
    
    test_batch_files(args.folder, args.num_files, args.output)