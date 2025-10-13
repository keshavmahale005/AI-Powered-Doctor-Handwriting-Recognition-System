import os
import cv2
import numpy as np
import json
import hashlib
from PIL import Image
import shutil

class ImageTrainer:
    """Class to handle training the system with specific image-text pairs"""
    
    def __init__(self, training_dir="training_data"):
        self.training_dir = training_dir
        self.database_file = os.path.join(training_dir, "training_database.json")
        self.images_dir = os.path.join(training_dir, "images")
        
        # Create directories if they don't exist
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Load or create the database
        if os.path.exists(self.database_file):
            with open(self.database_file, 'r') as f:
                self.database = json.load(f)
        else:
            self.database = {}
            self._save_database()
    
    def _calculate_image_hash(self, image_path):
        """Calculate a perceptual hash of an image for comparison"""
        try:
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to 32x32
            small = cv2.resize(gray, (32, 32))
            
            # Calculate the DCT of the image
            dct = cv2.dct(np.float32(small))
            
            # Take the top-left 8x8 of the DCT
            dct_low = dct[:8, :8]
            
            # Calculate the median value
            median = np.median(dct_low)
            
            # Create a hash from the DCT by comparing each value to the median
            hash_str = ''
            for i in range(8):
                for j in range(8):
                    hash_str += '1' if dct_low[i, j] > median else '0'
            
            # Additionally, create a content hash for exact matching
            img_bytes = cv2.imencode('.png', img)[1].tobytes()
            content_hash = hashlib.md5(img_bytes).hexdigest()
            
            return {
                "perceptual_hash": hash_str,
                "content_hash": content_hash
            }
            
        except Exception as e:
            print(f"Error calculating image hash: {str(e)}")
            return None
    
    def _hamming_distance(self, hash1, hash2):
        """Calculate the Hamming distance between two hash strings"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def _save_database(self):
        """Save the training database to file"""
        with open(self.database_file, 'w') as f:
            json.dump(self.database, f, indent=4)
    
    def add_training_sample(self, image_path, ocr_results):
        """Add a new training sample to the database"""
        # Calculate the image hash
        hash_data = self._calculate_image_hash(image_path)
        if not hash_data:
            return False
        
        # Create a unique filename for the saved image
        image_filename = f"{hash_data['content_hash']}{os.path.splitext(image_path)[1]}"
        saved_image_path = os.path.join(self.images_dir, image_filename)
        
        # Copy the image to our training data directory
        shutil.copy2(image_path, saved_image_path)
        
        # Add to database
        self.database[hash_data['content_hash']] = {
            "perceptual_hash": hash_data['perceptual_hash'],
            "image_path": saved_image_path,
            "ocr_results": ocr_results
        }
        
        # Save the updated database
        self._save_database()
        
        return True
    
    def find_match(self, image_path, similarity_threshold=5):
        """Find a match for the given image in the training database"""
        # Calculate the image hash
        hash_data = self._calculate_image_hash(image_path)
        if not hash_data:
            return None
        
        # First try exact content match
        if hash_data['content_hash'] in self.database:
            print(f"Exact content match found for {image_path}")
            return self.database[hash_data['content_hash']]['ocr_results']
        
        # If no exact match, try perceptual hash matching
        query_hash = hash_data['perceptual_hash']
        best_match = None
        best_distance = float('inf')
        
        for content_hash, entry in self.database.items():
            distance = self._hamming_distance(query_hash, entry['perceptual_hash'])
            if distance < best_distance:
                best_distance = distance
                best_match = entry
        
        # Return the match if it's below the similarity threshold
        if best_distance <= similarity_threshold and best_match:
            print(f"Similar image match found with distance {best_distance}")
            return best_match['ocr_results']
        
        return None

def main():
    # Example usage
    trainer = ImageTrainer()
    
    # Add a training sample
    sample_image = "path/to/sample.jpg"
    sample_results = {
        "raw_text": "This is a sample prescription.",
        "cleaned_text": "this is a sample prescription",
        "medications": ["aspirin", "paracetamol"],
        "confidence": 95.2
    }
    
    if os.path.exists(sample_image):
        success = trainer.add_training_sample(sample_image, sample_results)
        print(f"Added training sample: {success}")
    
    # Test finding a match
    test_image = "path/to/test.jpg"
    if os.path.exists(test_image):
        match = trainer.find_match(test_image)
        if match:
            print("Match found!")
            print(json.dumps(match, indent=4))
        else:
            print("No match found.")

if __name__ == "__main__":
    main()
