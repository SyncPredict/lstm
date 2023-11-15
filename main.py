from components.blockchain_data.get_data import fetchAllData
from components.create_model import create_model
from components.process_data import process_data
from components.test_model import test


def main():
    # 1 Fetch data
    data = fetchAllData()

    # 2 Process data
    processed_data = process_data(data)

    # 3 Creating model
    create_model(processed_data)

    # 4 Test model
    test(processed_data,'best_model.tf')


if __name__ == '__main__':
    main()
