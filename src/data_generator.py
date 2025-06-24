import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

class DataGenerator:
    """Generate sample datasets compatible with the data quality project."""
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Define regions and their cities
        self.regions_cities = {
            'North': ['New York', 'Boston', 'Philadelphia', 'Chicago'],
            'South': ['Houston', 'Dallas', 'San Antonio', 'Phoenix'],
            'East': ['Miami', 'Atlanta', 'Washington', 'Baltimore'],
            'West': ['Los Angeles', 'San Francisco', 'Seattle', 'San Diego']
        }
        
        # Flatten cities list
        self.all_cities = []
        for region, cities in self.regions_cities.items():
            self.all_cities.extend(cities)
        
        # Product categories and names
        self.product_categories = {
            'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera'],
            'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Shoes', 'Hat'],
            'Books': ['Fiction Novel', 'Technical Manual', 'Cookbook', 'Biography', 'Textbook'],
            'Home': ['Kitchen Set', 'Bedding', 'Lamp', 'Furniture', 'Decor'],
            'Sports': ['Basketball', 'Tennis Racket', 'Running Shoes', 'Gym Equipment', 'Bicycle']
        }
    
    def generate_regions_data(self, output_path=None):
        """Generate regions dataset with city mappings."""
        regions_data = []
        
        for region, cities in self.regions_cities.items():
            for city in cities:
                regions_data.append({
                    'city': city,
                    'region': region,
                    'country': 'United States',
                    'population': random.randint(500000, 5000000)
                })
        
        df = pd.DataFrame(regions_data)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✅ Generated regions data: {len(df)} rows -> {output_path}")
        
        return df
    
    def generate_customers_data(self, num_customers=1000, output_path=None):
        """Generate customers dataset."""
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Chris', 'Ashley', 
                      'Matthew', 'Jessica', 'Andrew', 'Amanda', 'Daniel', 'Melissa', 'James',
                      'Michelle', 'Robert', 'Lisa', 'William', 'Karen', 'Richard', 'Nancy',
                      'Joseph', 'Betty', 'Thomas', 'Helen', 'Charles', 'Sandra', 'Christopher']
        
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
                     'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
                     'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
                     'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark']
        
        customers_data = []
        
        for i in range(1, num_customers + 1):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            
            # Generate email with some invalid ones for testing
            if random.random() < 0.05:  # 5% invalid emails
                email = f"invalid-email-{i}"
            else:
                email = f"{first_name.lower()}.{last_name.lower()}{i}@example.com"
            
            # Generate age with some invalid ones for testing
            if random.random() < 0.02:  # 2% invalid ages
                age = random.choice([-1, -5, 150, 200])
            else:
                age = random.randint(18, 80)
            
            customers_data.append({
                'customer_id': i,
                'name': f"{first_name} {last_name}",
                'email': email,
                'age': age,
                'city': random.choice(self.all_cities),
                'registration_date': self._random_date('2023-01-01', '2025-06-30')
            })
        
        # Add some duplicates for testing
        if num_customers > 10:
            duplicate_customer = customers_data[5].copy()
            customers_data.append(duplicate_customer)
        
        df = pd.DataFrame(customers_data)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✅ Generated customers data: {len(df)} rows -> {output_path}")
        
        return df
    
    def generate_products_data(self, num_products=500, output_path=None):
        """Generate products dataset."""
        products_data = []
        
        for i in range(1, num_products + 1):
            category = random.choice(list(self.product_categories.keys()))
            product_name = random.choice(self.product_categories[category])
            
            # Add variation to product names
            if random.random() < 0.3:
                variations = ['Pro', 'Plus', 'Deluxe', 'Standard', 'Premium', 'Basic']
                product_name += f" {random.choice(variations)}"
            
            products_data.append({
                'product_id': i,
                'product_name': product_name,
                'category': category,
                'price': round(random.uniform(10.0, 500.0), 2),
                'stock_quantity': random.randint(0, 1000)
            })
        
        df = pd.DataFrame(products_data)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✅ Generated products data: {len(df)} rows -> {output_path}")
        
        return df
    
    def generate_orders_data(self, customers_df, products_df, num_orders=2000, output_path=None):
        """Generate orders dataset."""
        orders_data = []
        
        customer_ids = customers_df['customer_id'].tolist()
        product_ids = products_df['product_id'].tolist()
        
        for i in range(1, num_orders + 1):
            customer_id = random.choice(customer_ids)
            product_id = random.choice(product_ids)
            quantity = random.randint(1, 10)
            
            # Get product price
            product_price = products_df[products_df['product_id'] == product_id]['price'].iloc[0]
            total_amount = round(quantity * product_price, 2)
            
            # Generate order date with concentration in certain months for testing
            if random.random() < 0.4:  # 40% in target months
                order_date = self._random_date('2024-01-01', '2024-01-31')
            elif random.random() < 0.3:  # 30% in another month
                order_date = self._random_date('2024-02-01', '2024-02-28')
            else:  # Rest spread across other dates
                order_date = self._random_date('2023-06-01', '2025-06-30')
            
            orders_data.append({
                'order_id': i,
                'customer_id': customer_id,
                'product_id': product_id,
                'quantity': quantity,
                'order_date': order_date,
                'total_amount': total_amount
            })
        
        df = pd.DataFrame(orders_data)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✅ Generated orders data: {len(df)} rows -> {output_path}")
        
        return df
    
    def _random_date(self, start_date, end_date):
        """Generate random date between start_date and end_date."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        time_between = end - start
        days_between = time_between.days
        random_days = random.randrange(days_between)
        
        return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')
    
    def generate_all_datasets(self, data_dir='data', num_customers=1000, num_products=500, num_orders=2000):
        """Generate all datasets and save to files."""
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)
        
        print("🚀 Generating compatible datasets for data quality project...")
        print("="*60)
        
        # Generate regions
        regions_df = self.generate_regions_data(data_path / 'regions.csv')
        
        # Generate customers
        customers_df = self.generate_customers_data(num_customers, data_path / 'customers.csv')
        
        # Generate products
        products_df = self.generate_products_data(num_products, data_path / 'products.csv')
        
        # Generate orders
        orders_df = self.generate_orders_data(customers_df, products_df, num_orders, data_path / 'orders.csv')
        
        print("\n" + "="*60)
        print("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Data saved to: {data_path.absolute()}")
        
        # Print summary statistics
        print("\n📊 DATASET SUMMARY:")
        print(f"   🏢 Regions: {len(regions_df)} cities across {regions_df['region'].nunique()} regions")
        print(f"   👥 Customers: {len(customers_df)} customers")
        print(f"   📦 Products: {len(products_df)} products across {products_df['category'].nunique()} categories")
        print(f"   🛒 Orders: {len(orders_df)} orders")
        
        # Show region distribution
        print(f"\n🗺️  REGION DISTRIBUTION:")
        region_customer_count = customers_df.merge(regions_df, on='city')['region'].value_counts()
        for region, count in region_customer_count.items():
            print(f"   {region}: {count} customers")
        
        # Show date range
        print(f"\n📅 ORDER DATE RANGE:")
        print(f"   From: {orders_df['order_date'].min()}")
        print(f"   To: {orders_df['order_date'].max()}")
        
        # Show monthly distribution
        orders_df['order_month'] = pd.to_datetime(orders_df['order_date']).dt.to_period('M')
        monthly_counts = orders_df['order_month'].value_counts().head(5)
        print(f"\n📈 TOP MONTHS BY ORDER COUNT:")
        for month, count in monthly_counts.items():
            print(f"   {month}: {count} orders")
        
        return {
            'regions': regions_df,
            'customers': customers_df,
            'products': products_df,
            'orders': orders_df
        }

def main():
    """Main function to generate datasets."""
    print("🔧 Data Generator for Data Quality Project")
    print("="*50)
    
    generator = DataGenerator(seed=42)
    
    # Generate datasets
    datasets = generator.generate_all_datasets(
        data_dir='data',
        num_customers=1000,
        num_products=500,
        num_orders=2000
    )
    
    print("\n💡 CONFIGURATION RECOMMENDATIONS:")
    print("Update your config/pipeline_config.yaml with:")
    print("```yaml")
    print("pipeline:")
    print("  input_path: 'data/'")
    print("  output_path: 'output/'")
    print("  target_month: '2024-01'  # High concentration of orders")
    print("  region_filters: ['North', 'South', 'East', 'West']")
    print("```")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run: python src/data_generator.py")
    print("2. Update your pipeline_config.yaml")
    print("3. Run: python src/main.py")

if __name__ == "__main__":
    main()