from statmerge.metrics_mod.agreement import binary_metrics, fleiss_kappa_from_raw, icc2_1
def test_binary_metrics_counts():
    y=[1,1,0,0,1,0,1,0]; p=[1,0,0,0,1,1,1,0]
    m=binary_metrics(y,p)
    assert m['N']==8 and 'Accuracy' in m
def test_fleiss_and_icc_shapes():
    M=[['A','A','B'],['B','B','B'],['A','B','B'],['A','A','A']]
    k,_,_=fleiss_kappa_from_raw(M)
    assert k==k
    D=[[1.2,1.1,0.9],[1.0,0.8,0.7],[1.3,1.2,1.0],[0.9,0.7,0.6]]
    v=icc2_1(D)
    assert v==v
