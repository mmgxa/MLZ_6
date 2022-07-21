import pickle
import pandas as pd
import os
import boto3
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL') # only for writin predictions
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

options = {
'client_kwargs': {
    'endpoint_url': S3_ENDPOINT_URL
    }
}
    
def get_input_path(year, month):
    default_input_pattern = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/fhv-{year:04d}-{month:02d}-predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def push_data(month, year):
    if os.getenv('INPUT_FILE_PATTERN'):
        data = [
                (None, None, dt(1, 2), dt(1, 10)),
                (1, 1, dt(1, 2), dt(1, 10)),
                (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
                (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
            ]
        columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
        df = pd.DataFrame(data, columns=columns)
        df.to_parquet(
            f's3://nyc-duration/fhv-{year:04d}-{month:02d}-input.parquet',
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
            )
    

def read_data(filename):
    if S3_ENDPOINT_URL:
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)
    
    return df

def prepare_data(df, categorical):
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(month: str, year: str):
    categorical = ['PUlocationID', 'DOlocationID']

    input_file = get_input_path(year, month)
        
    # output_file = get_output_path(year, month)
    df = read_data(input_file)
    if S3_ENDPOINT_URL:
        # if endpoint exists
        
        # write to that bucket
        df.to_parquet(
            f's3://nyc-duration/fhv-{year:04d}-{month:02d}-input.parquet',
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
            )
    else:
        input_file_path = f'in/fhv-{year:04d}-{month:02d}-input.parquet'
        # write to local filesystem
        df.to_parquet(
            input_file_path,
            engine='pyarrow',
            compression=None,
            index=False,
            )
    
    df = prepare_data(df, categorical)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    
    with open('model/model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted sum of duration:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    
    if S3_ENDPOINT_URL:
        # if endpoint exists
        # write to that bucket
        df_result.to_parquet(
            get_output_path(year, month),
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
            )
    else:
        # local output file path if no endpoint found
        output_file = f'out/fhv-{year:04d}-{month:02d}-predictions.parquet'
        # write to local filesystem
        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False,
            )
    
    return {'predicted sum of duration': y_pred.sum()} 
    
if __name__ == "__main__":
    # always create bucket
    s3_client = boto3.client('s3', endpoint_url=S3_ENDPOINT_URL, region_name=AWS_DEFAULT_REGION)
    s3_client.create_bucket(Bucket='nyc-duration', CreateBucketConfiguration={
    'LocationConstraint': AWS_DEFAULT_REGION})
    year = int(os.getenv('YEAR'))
    month = int(os.getenv('MONTH'))
    # push fake data if INPUT_FILE_PATTERN is specified
    push_data(month, year)
    main(month, year)
