def test_derivative():
    from nnlibrary.differentiators import Variable

    def op_addition():
        x = Variable(3)
        y = Variable(5)

        # z = x + y
        target = 1

        # forward pass
        x.dot = 1
        y.dot = 0
        z = x + y
        value = z.dot

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert target == value, error_prompt

    def op_difference():
        x = Variable(3)
        y = Variable(5)

        # z = y - x
        target = -1

        # forward pass
        x.dot = 1
        y.dot = 0
        z = y - x
        value = z.dot

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert target == value, error_prompt

    def op_negative():
        x = Variable(3)

        # z = - x
        target = -1

        # forward pass
        x.dot = 1
        z = - x
        value = z.dot

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert target == value, error_prompt

    def op_positive():
        x = Variable(3)

        # z = x
        target = 1

        # forward pass
        x.dot = 1
        z = x
        value = z.dot

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert target == value, error_prompt

    def op_multiplication():
        x = Variable(3)
        y = Variable(5)

        # z = x * y
        target = y.value

        # forward pass
        x.dot = 1
        y.dot = 0
        z = x * y
        value = z.dot

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert target == value, error_prompt

    def op_division():
        x = Variable(3)
        y = Variable(5)

        # z = y / x
        target = -y.value / x.value ** 2

        # forward pass
        x.dot = 1
        y.dot = 0
        z = y / x
        value = z.dot

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert target == value, error_prompt

    def op_power():
        x = Variable(3)
        y = Variable(5)

        # z = x ** y
        target = y.value * x.value ** (y.value - 1)

        # forward pass
        x.dot = 1
        y.dot = 0
        z = x ** y
        value = z.dot

        error_prompt = f'\n  Target and Value are not the same: \n' \
                       f'    Target: {target}\n' \
                       f'    Value : {value}'

        assert target == value, error_prompt

    # op_addition()
    # op_difference()
    # op_negative()
    # op_positive()
    # op_multiplication()
    # op_division()
    # op_power()

    from nnlibrary.differentiators import Graph

    with Graph() as g:
        x = Variable(2)
        y = Variable(3)
        c = Variable(5)
        z = x * y + c

    print(g)

    with Graph() as g:
        a = Variable(2)
        b = Variable(3)
        d = Variable(5)
        e = a * b + d

    print(g.variables)
    exit(-2)


