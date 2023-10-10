rup_A = {"idx": "A", "D": 1.0, "M": 6.0, "faults": ["f1"]}

rup_B = {"idx": "B", "D": 1.0, "M": 6.0, "faults": ["f2"]}

rup_C = {"idx": "C", "D": 2.5, "M": 7.0, "faults": ["f1", "f2"]}

rup_D = {"idx": "D", "D": 1.75, "M": 6.5, "faults": ["f1", "f2"]}

f1 = {"id": "f1", "slip_rate": 1.0, "slip_rate_err": 1.0}

f2 = {"id": "f2", "slip_rate": 1.0, "slip_rate_err": 1.0}


simple_test_rups = [rup_A, rup_B, rup_C, rup_D]
simple_test_faults = [f1, f2]

simple_test_fault_adjacence = {0: [1], 1: [0]}
