function validateGroupStructure(v_groups)
    assert(all(v_groups==abs(round(v_groups))), ...
        'All entries in v_group_structure must be natural numbers')
    assert(min(v_groups)==1, 'Group labels must start by 1')
    for gr = 1:max(v_groups)
        assert(sum(v_groups==gr)>0, ...
            'Groups must be labeled with consecutive numbers')
    end
end