from statmerge.analysis.effects import cohen_d, hedges_g
def test_effects_basic():
    a=[1,2,3,4,5]; b=[2,3,4,5,6]
    d=cohen_d(a,b); g=hedges_g(a,b)
    assert d!=0 and g!=0
