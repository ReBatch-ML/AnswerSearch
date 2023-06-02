"""script to deploy and delete all backend services
"""
import argparse
import deploy_dolly_backend, deploy_milvus_backend, deploy_ss_backend, deploy_flan_backend
from pymilvus import connections, Collection
import time

def start_all():
    """start all backend services
    """
    try:
        print("Starting Milvus") 
        deploy_milvus_backend.start_milvus()
    except Exception as e:
        print("Could not start Milvus.")
        print(e )
    is_connected = False
    while(not is_connected):
        try:
            connections.connect(
            alias="default",
            user='username',
            password='password',
            host='20.50.24.59',
            port='19530'
            )
            print("Milvus server is connected.")
            collection = Collection("milvus_vectors")      # Get an existing collection.

            print("Collection found with ",collection.num_entities, " entities.")
            is_connected = True
        except Exception as e:
            print("Milvus server is not yet running.")
            print(e )
            time.sleep(30)


    try:
        print("Starting SemSearch")
        deploy_ss_backend.deploy()
    except Exception as e:
        print("Could not start SemSearch.")
        print(e )

    try:
        print("Starting Flan")
        deploy_flan_backend.deploy()
    except Exception as e:
        print("Could not start Flan.")
        print(e )

    try:
        print("Starting Dolly")
        deploy_dolly_backend.deploy()
    except Exception as e:
        print("Could not start Dolly.")
        print(e )


def stop_all():
    """stop all backend services"""
    try:
        print("Stopping Dolly")
        deploy_dolly_backend.delete()
    except Exception as e:
        print("Could not stop Dolly.")
        print(e )
    try:
        print("Stopping Flan")
        deploy_flan_backend.delete()
    except Exception as e:
        print("Could not stop Flan.")
        print(e )
    
    try:
        print("Stopping SemSearch")
        deploy_ss_backend.delete()
    except Exception as e:
        print("Could not stop SemSearch.")
        print(e )
    

    try:
        print("Stopping Milvus")
        deploy_milvus_backend.stop_milvus()
    except Exception as e:
        print("Could not stop Milvus.")
        print(e )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arg1", nargs='?', default="start")
    args = parser.parse_args()
    if args.arg1 == "start":
        start_all()
    elif args.arg1 == "stop":
        stop_all()
    else:
        print("Invalid argument")
