for d in mutag ptc_mm enzymes proteins nci109 nci1
do


    python run_tu.py --seed=0 --pci_id=0 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=1 --pci_id=0 -c=configs/config.json -d=$d &
    python run_tu.py --seed=2 --pci_id=1 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=3 --pci_id=1 -c=configs/config.json -d=$d &
    python run_tu.py --seed=4 --pci_id=2 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=5 --pci_id=2 -c=configs/config.json -d=$d &


     for job in `jobs -p`
        do
        echo $job
            wait $job 
        done


    python run_tu.py --seed=6 --pci_id=0 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=7 --pci_id=0 -c=configs/config.json -d=$d &
    python run_tu.py --seed=8 --pci_id=1 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=9 --pci_id=1 -c=configs/config.json -d=$d &
    python run_tu.py --seed=10 --pci_id=2 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=11 --pci_id=2 -c=configs/config.json -d=$d &


     for job in `jobs -p`
        do
        echo $job
            wait $job 
        done


    python run_tu.py --seed=12 --pci_id=0 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=13 --pci_id=0 -c=configs/config.json -d=$d &
    python run_tu.py --seed=14 --pci_id=1 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=15 --pci_id=1 -c=configs/config.json -d=$d &
    python run_tu.py --seed=16 --pci_id=2 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=17 --pci_id=2 -c=configs/config.json -d=$d &


     for job in `jobs -p`
        do
        echo $job
            wait $job 
        done


    python run_tu.py --seed=18 --pci_id=0 -c=configs/config.json -d=$d &  
    python run_tu.py --seed=19 --pci_id=1 -c=configs/config.json -d=$d &
    python run_tu.py --seed=20 --pci_id=2 -c=configs/config.json -d=$d &


     for job in `jobs -p`
        do
        echo $job
            wait $job 
        done
done
