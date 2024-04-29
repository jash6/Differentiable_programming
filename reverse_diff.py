import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random

# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def reverse_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_rev : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def accum_deriv(target, deriv, overwrite):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    return [loma_ir.Assign(target,
                        loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False
    
    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def check_lhs_is_output_args(lhs, output_args):
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_args
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_args(lhs.array, output_args)
            case loma_ir.StructAccess():
                return check_lhs_is_output_args(lhs.struct, output_args)
            case _:
                assert False
    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(\
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(\
                val,
                lineno = node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(\
                node.target,
                node.t,
                val,
                lineno = node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Assign(\
                target,
                val,
                lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(\
                call,
                lineno = node.lineno)]

        def mutate_call(self, node):
            new_args = []
            for arg in node.args:
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(\
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(\
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t = node.t)

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.

    class FwdPassMutator(irmutator.IRMutator):
        total = 0
        def __init__(self, output_args=None):
            self.output_args = output_args

        def mutate_function_def(self, node):
            self.declare_stmts = []
            self.count = 0
            for stmts in range(len(node.body)):
                if isinstance(node.body[stmts], loma_ir.Assign):
                    self.count +=1

            self.array = loma_ir.Declare('_t_float',loma_ir.Array(loma_ir.Float(),self.count))
            self.declare_stmts.append(self.array)
            self.pointer_float = loma_ir.Declare('_stack_ptr_float', loma_ir.Int())
            self.declare_stmts.append(self.pointer_float)

            self.array2 = loma_ir.Declare('_t_int',loma_ir.Array(loma_ir.Int(),self.count))
            self.declare_stmts.append(self.array2)
            self.pointer_int = loma_ir.Declare('_stack_ptr_int', loma_ir.Int())
            self.declare_stmts.append(self.pointer_int)

            for stmt in range(len(node.body)):
                if isinstance(node.body[stmt], loma_ir.Assign):
                    if not check_lhs_is_output_args(node.body[stmt].target, self.output_args):
                        if isinstance(node.body[stmt].target, loma_ir.StructAccess):
                            if isinstance(node.body[stmt].target.t, loma_ir.Float):
                                self.declare_stmts.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var('_t_float'), loma_ir.Var('_stack_ptr_float')), loma_ir.StructAccess(loma_ir.Var(node.body[stmt].target.struct.id),node.body[stmt].target.member_id)))
                                self.declare_stmts.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_float'), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var('_stack_ptr_float'), loma_ir.ConstInt(1))))
                            elif isinstance(node.body[stmt].target.t, loma_ir.Int):
                                self.declare_stmts.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var('_t_int'), loma_ir.Var('_stack_ptr_int')), loma_ir.StructAccess(loma_ir.Var(node.body[stmt].target.struct.id),node.body[stmt].target.member_id)))
                                self.declare_stmts.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_int'), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var('_stack_ptr_int'), loma_ir.ConstInt(1))))
                        elif isinstance(node.body[stmt].target.t, loma_ir.Float):
                            self.declare_stmts.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var('_t_float'), loma_ir.Var('_stack_ptr_float')), loma_ir.Var(node.body[stmt].target.id)))
                            self.declare_stmts.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_float'), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var('_stack_ptr_float'), loma_ir.ConstInt(1))))
                        elif isinstance(node.body[stmt].target.t, loma_ir.Int):
                            self.declare_stmts.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var('_t_int'), loma_ir.Var('_stack_ptr_int')), loma_ir.Var(node.body[stmt].target.id)))
                            self.declare_stmts.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_int'), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var('_stack_ptr_int'), loma_ir.ConstInt(1))))
                        self.declare_stmts.append(node.body[stmt])
                        self.mutate_stmt(node.body[stmt])

                elif isinstance(node.body[stmt], loma_ir.Declare):
                    self.declare_stmts.append(node.body[stmt])
                    self.mutate_stmt(node.body[stmt])
                    if node.body[stmt].val is not None:
                        if isinstance(node.body[stmt].t, loma_ir.Float):
                            self.declare_stmts.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var('_t_float'), loma_ir.Var('_stack_ptr_float')), loma_ir.Var(node.body[stmt].target)))
                            self.declare_stmts.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_float'), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var('_stack_ptr_float'), loma_ir.ConstInt(1))))
                        elif isinstance(node.body[stmt].t, loma_ir.Int):
                            self.declare_stmts.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var('_t_int'), loma_ir.Var('_stack_ptr_int')), loma_ir.Var(node.body[stmt].target)))
                            self.declare_stmts.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_int'), loma_ir.BinaryOp(loma_ir.Add(), loma_ir.Var('_stack_ptr_int'), loma_ir.ConstInt(1))))
        
        def mutate_declare(self, node):
            self.total += 1
            if isinstance(node.t, loma_ir.Float):
                stmt = loma_ir.Declare('_d'+node.target, node.t)
                self.declare_stmts.append(stmt)
                return
            elif isinstance(node.t, loma_ir.Int):
                return
            elif isinstance(node.t, loma_ir.Struct):
                self.declare_stmts.append(loma_ir.Declare('_d'+node.target, node.t))
                return
        
        def mutate_assign(self, node):
            if isinstance(node.target, loma_ir.Var):
                if isinstance(node.target.t, loma_ir.Float):
                    stmt = loma_ir.Assign(loma_ir.Var('_d'+node.target.id), loma_ir.ConstFloat(0.0))
                    self.declare_stmts.append(stmt)
                elif isinstance(node.target.t, loma_ir.Int):
                    return
                return
            elif isinstance(node.target, loma_ir.ArrayAccess):
                stmt = loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var('_d'+node.target.array.id), node.target.index), loma_ir.ConstFloat(0.0))
                self.declare_stmts.append(stmt)
                return
            
            elif isinstance(node.target, loma_ir.StructAccess):
                if node.target.t == loma_ir.Float():
                    stmt = loma_ir.Assign(loma_ir.StructAccess(loma_ir.Var('_d'+node.target.struct.id), node.target.member_id, t = node.target.t), loma_ir.ConstFloat(0.0))
                    self.declare_stmts.append(stmt)
                    return
                elif node.target.t == loma_ir.Int():
                    return
        
        def mutate_var(self, node):
            self.total += 1
            return super().mutate_var(node)

    # Apply the differentiation.
    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.counter = 0
            self.list1 = []
            self.flag_switch = 0
            self.counter2 = 0
            self.list2 = []
            new_args = []
            self.output_args = [arg.id for arg in node.args if arg.i == loma_ir.Out()]
            for arg in node.args:
                if arg.i == loma_ir.In():
                    new_args.append(arg)
                    new_arg_id = '_d'+arg.id
                    new_args.append(loma_ir.Arg(new_arg_id, arg.t, loma_ir.Out()))
                if arg.i == loma_ir.Out():
                    new_args.append(loma_ir.Arg(arg.id,arg.t, loma_ir.In()))
            if node.ret_type is not None:
                new_args.append(loma_ir.Arg('_dreturn', node.ret_type, loma_ir.In()))  
            
            
            fm = FwdPassMutator(self.output_args)
            fm.mutate_function_def(node)
            new_body_fwd = fm.declare_stmts
            new_body_fwd = irmutator.flatten(new_body_fwd)

            
            self.tape = fm.array
            self.ptr_position = fm.pointer_float
            self.int_tape = fm.array2
            self.int_position = fm.pointer_int

            
            new_body_rev = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]
            new_body_rev = irmutator.flatten(new_body_rev)
            new_body = new_body_fwd + new_body_rev

            new_func = loma_ir.FunctionDef(diff_func_id, new_args, new_body, node.is_simd, None)

            return new_func

        def mutate_return(self, node):
            if isinstance(node.val.t, loma_ir.Struct):
                self.adjoint = loma_ir.Var('_dreturn', t = node.val.t)
                dervi = loma_ir.Var('_d'+node.val.id, t = node.val.t)
                stmts = accum_deriv(dervi, self.adjoint, False)
                self.adjoint = None
                return stmts
            else:
                self.adjoint = loma_ir.Var('_dreturn')
                stmts1 = self.mutate_expr(node.val)
                self.adjoint = None
                self.flag_switch = 1

                self.adjoint = loma_ir.Var('_dreturn')
                stmts2 = self.mutate_expr(node.val)
                self.adjoint = None
                self.flag_switch = 0
                return stmts1 + stmts2

        def mutate_declare(self, node):
            if node.val is not None:
                if isinstance(node.t, loma_ir.Struct):
                    self.adjoint = loma_ir.Var(f'_d{node.target}', t = node.t)
                    deriv = loma_ir.Var(f'_d{node.val.id}', t = node.t)
                    stmts = accum_deriv(deriv, self.adjoint, False)
                    self.adjoint = None
                    return stmts
                else:
                    if isinstance(node.t, loma_ir.Float):
                        self.adjoint = loma_ir.Var(f'_d{node.target}')
                        stmts1 = self.mutate_expr(node.val)
                        self.adjoint = None
                        self.flag_switch = 1

                        stmts1.append(loma_ir.Assign(loma_ir.Var('_d'+ node.target), loma_ir.ConstFloat(0.0)))

                        self.adjoint = loma_ir.Var(f'_d{node.target}')
                        stmts2 = self.mutate_expr(node.val)
                        self.adjoint = None
                        self.flag_switch = 0
                        return stmts1 + stmts2
            return []

        def mutate_assign(self, node):
            tape_stmt = []

            if not check_lhs_is_output_args(node.target, self.output_args):
                if isinstance(node.target, loma_ir.StructAccess):
                    if isinstance(node.target.t, loma_ir.Float):
                        tape_stmt.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_float'), loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var('_stack_ptr_float'), loma_ir.ConstInt(1))))
                        tape_stmt.append(loma_ir.Assign(loma_ir.StructAccess(loma_ir.Var(node.target.struct.id),node.target.member_id), loma_ir.ArrayAccess(loma_ir.Var('_t_float'), loma_ir.Var('_stack_ptr_float'))))
                    elif isinstance(node.target.t, loma_ir.Int):
                        tape_stmt.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_int'), loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var('_stack_ptr_int'), loma_ir.ConstInt(1))))
                        tape_stmt.append(loma_ir.Assign(loma_ir.StructAccess(loma_ir.Var(node.target.struct.id),node.target.member_id), loma_ir.ArrayAccess(loma_ir.Var('_t_int'), loma_ir.Var('_stack_ptr_int'))))
                elif isinstance(node.target.t, loma_ir.Float):
                    tape_stmt.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_float'), loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var('_stack_ptr_float'), loma_ir.ConstInt(1))))
                    tape_stmt.append(loma_ir.Assign(loma_ir.Var(node.target.id), loma_ir.ArrayAccess(loma_ir.Var('_t_float'), loma_ir.Var('_stack_ptr_float'))))

                elif isinstance(node.target.t, loma_ir.Int):
                    tape_stmt.append(loma_ir.Assign(loma_ir.Var('_stack_ptr_int'), loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.Var('_stack_ptr_int'), loma_ir.ConstInt(1))))
                    tape_stmt.append(loma_ir.Assign(loma_ir.Var(node.target.id), loma_ir.ArrayAccess(loma_ir.Var('_t_int'), loma_ir.Var('_stack_ptr_int'))))

            if check_lhs_is_output_args(node.target, self.output_args):
                if isinstance(node.target, loma_ir.Var):
                    self.adjoint = loma_ir.Var(node.target.id)
                    stmts1 = self.mutate_expr(node.val)
                    self.adjoint = None
                    self.flag_switch = 1

                    self.adjoint = loma_ir.Var(node.target.id)
                    stmts2 = self.mutate_expr(node.val)
                    self.adjoint = None
                    self.flag_switch = 0
                    return tape_stmt + stmts1 + stmts2
                elif isinstance(node.target, loma_ir.ArrayAccess):
                    self.adjoint = loma_ir.ArrayAccess(loma_ir.Var(node.target.array.id), node.target.index)
                    stmts1 = self.mutate_expr(node.val)
                    self.adjoint = None
                    self.flag_switch = 1

                    self.adjoint = loma_ir.ArrayAccess(loma_ir.Var(node.target.array.id), node.target.index)
                    stmts2 = self.mutate_expr(node.val)
                    self.adjoint = None
                    self.flag_switch = 0
                    return tape_stmt + stmts1 + stmts2
                
            if isinstance(node.target, loma_ir.Var):
                if isinstance(node.target.t, loma_ir.Struct):
                    self.adjoint = loma_ir.Var(f'_d{node.target.id}')
                    stmts1 = self.mutate_expr(node.val)
                    self.adjoint = None
                    return tape_stmt + stmts1

                else:
                    self.adjoint = loma_ir.Var(f'_d{node.target.id}')
                    stmts1 = self.mutate_expr(node.val)
                    self.adjoint = None
                    self.flag_switch = 1

                    if isinstance(node.target.t, loma_ir.Float):
                        stmts1.append(loma_ir.Assign(loma_ir.Var('_d'+ node.target.id), loma_ir.ConstFloat(0.0)))

                    self.adjoint = loma_ir.Var(f'_d{node.target.id}')
                    stmts2 = self.mutate_expr(node.val)
                    self.adjoint = None
                    self.flag_switch = 0
                    return tape_stmt + stmts1 + stmts2
            
            elif isinstance(node.target, loma_ir.ArrayAccess):
                self.adjoint = loma_ir.ArrayAccess(loma_ir.Var(f'_d{node.target.array.id}'), node.target.index)
                stmts1 = self.mutate_expr(node.val)
                self.adjoint = None
                self.flag_switch = 1

                if isinstance(node.target.array.t, loma_ir.Float):
                    stmts1.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var('_d'+node.target.array.id), node.target.index), loma_ir.ConstFloat(0.0)))

                self.adjoint = loma_ir.ArrayAccess(loma_ir.Var(f'_d{node.target.array.id}'), node.target.index)
                stmts2 = self.mutate_expr(node.val)
                self.adjoint = None
                self.flag_switch = 0
                return tape_stmt + stmts1 + stmts2
            
            elif isinstance(node.target, loma_ir.StructAccess):
                self.adjoint = loma_ir.StructAccess(loma_ir.Var(f'_d{node.target.struct.id}'), node.target.member_id, t = node.target.t)
                stmts1 = self.mutate_expr(node.val)
                self.adjoint = None
                self.flag_switch = 1
                
                if isinstance(node.target.t, loma_ir.Float):
                    stmts1.append(loma_ir.Assign(loma_ir.StructAccess(loma_ir.Var('_d'+node.target.struct.id), node.target.member_id, t = node.target.t), loma_ir.ConstFloat(0.0)))
                
                self.adjoint = loma_ir.StructAccess(loma_ir.Var(f'_d{node.target.struct.id}'), node.target.member_id, t = node.target.t)
                stmts2 = self.mutate_expr(node.val)
                self.adjoint = None
                self.flag_switch = 0

                return tape_stmt + stmts1 + stmts2

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_call_stmt(self, node):
            # HW3: TODO
            return super().mutate_call_stmt(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # HW2: TODO
            return []

        def mutate_const_int(self, node):
            # HW2: TODO
            return []

        def mutate_var(self, node):
            # HW2: TODO
            
            if isinstance(node.t, loma_ir.Int):

                if self.flag_switch == 0:
                    self.list2.append(self.counter2)
                    self.counter2 += 1
                if self.flag_switch == 1:
                    return []

                return []
            
            if isinstance(node.t, loma_ir.Struct):
                stmts = []
                stmts.append(loma_ir.Assign(loma_ir.Var('_d'+node.id),self.adjoint))
                return stmts
            
            stmts = []
            if self.flag_switch == 0:
                stmts.append(loma_ir.Declare('adj'+str(self.counter), loma_ir.Float()))
                stmts.append(loma_ir.Assign(loma_ir.Var('adj'+str(self.counter)), self.adjoint))
                self.list1.append(self.counter)
                self.counter += 1
            if self.flag_switch == 1:
                stmts.append(loma_ir.Assign(loma_ir.Var('_d'+node.id),loma_ir.BinaryOp(loma_ir.Add(),loma_ir.Var('_d'+node.id), loma_ir.Var('adj'+str(self.list1.pop(0))))))
            
            return stmts

        def mutate_array_access(self, node):
            # HW2: TODO
            if isinstance(node.array, loma_ir.ArrayAccess):
                stmts = []
                if self.flag_switch == 0:
                    stmts.append(loma_ir.Declare('adj'+str(self.counter), loma_ir.Float()))
                    stmts.append(loma_ir.Assign(loma_ir.Var('adj'+str(self.counter)), self.adjoint))
                    self.list1.append(self.counter)
                    self.counter += 1
                if self.flag_switch == 1:
                    stmts.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.ArrayAccess(loma_ir.Var('_d'+node.array.array.id),node.array.index), node.index),
                                                loma_ir.BinaryOp(loma_ir.Add(),
                                                                 loma_ir.ArrayAccess(loma_ir.ArrayAccess(loma_ir.Var('_d'+node.array.array.id),node.array.index), node.index), 
                                                                 loma_ir.Var('adj'+str(self.list1.pop(0))))))
                    
                return stmts

            if isinstance(node.array.t, loma_ir.Int):
                if self.flag_switch == 0:
                    self.list2.append(self.counter2)
                    self.counter2 += 1
                if self.flag_switch == 1:
                    return []

                return []
            
            stmts = []
            if self.flag_switch == 0:
                stmts.append(loma_ir.Declare('adj'+str(self.counter), loma_ir.Float()))
                stmts.append(loma_ir.Assign(loma_ir.Var('adj'+str(self.counter)), self.adjoint))
                self.list1.append(self.counter)
                self.counter += 1
            if self.flag_switch == 1:
                stmts.append(loma_ir.Assign(loma_ir.ArrayAccess(loma_ir.Var('_d'+node.array.id), node.index),loma_ir.BinaryOp(loma_ir.Add(),loma_ir.ArrayAccess(loma_ir.Var('_d'+node.array.id), node.index), loma_ir.Var('adj'+str(self.list1.pop(0))))))
            
            return stmts

        def mutate_struct_access(self, node):
            if isinstance(node.t, loma_ir.Int):
                if self.flag_switch == 0:
                    self.list2.append(self.counter2)
                    self.counter2 += 1
                if self.flag_switch == 1:
                    return []
                return []
            stmts = []
            if self.flag_switch == 0:
                stmts.append(loma_ir.Declare('adj'+str(self.counter), loma_ir.Float()))
                stmts.append(loma_ir.Assign(loma_ir.Var('adj'+str(self.counter)), self.adjoint))
                self.list1.append(self.counter)
                self.counter += 1
            if self.flag_switch == 1:
                stmts.append(loma_ir.Assign(loma_ir.StructAccess(loma_ir.Var('_d'+node.struct.id), node.member_id, t = node.t),loma_ir.BinaryOp(loma_ir.Add(), loma_ir.StructAccess(loma_ir.Var('_d'+node.struct.id), node.member_id, t = node.t),loma_ir.Var('adj'+str(self.list1.pop(0))))))
            
            return stmts

        def mutate_add(self, node):
            stmts_left = self.mutate_expr(node.left)
            stmts_right = self.mutate_expr(node.right)
            return stmts_left + stmts_right

        def mutate_sub(self, node):
            stmts_left = self.mutate_expr(node.left)
            old_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), self.adjoint)
            stmts_right = self.mutate_expr(node.right)
            self.adjoint = old_adjoint
            return stmts_left + stmts_right

        def mutate_mul(self, node):
            old_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, node.right)
            stmts_left = self.mutate_expr(node.left)
            self.adjoint = old_adjoint


            old_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, node.left)
            stmts_right = self.mutate_expr(node.right)
            self.adjoint = old_adjoint

            return stmts_left + stmts_right

        def mutate_div(self, node):
            old_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), self.adjoint, node.right)
            stmts_left = self.mutate_expr(node.left)
            self.adjoint = old_adjoint


            old_adjoint = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), self.adjoint)
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, node.left)
            self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), self.adjoint, loma_ir.BinaryOp(loma_ir.Mul(), node.right, node.right))
            stmts_right = self.mutate_expr(node.right)
            self.adjoint = old_adjoint

            return stmts_left + stmts_right

        def mutate_call(self, node):
            if node.id == 'sin':
                old_adjoint = self.adjoint
                self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, loma_ir.Call('cos',node.args))
                stmts_left = self.mutate_expr(node.args[0])
                self.adjoint = old_adjoint
                return stmts_left

            if node.id == 'cos':
                old_adjoint = self.adjoint
                self.adjoint = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), self.adjoint)
                self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, loma_ir.Call('sin',node.args))
                stmts_left = self.mutate_expr(node.args[0])
                self.adjoint = old_adjoint
                return stmts_left
            
            if node.id == 'sqrt':
                old_adjoint = self.adjoint
                self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), self.adjoint, loma_ir.BinaryOp(loma_ir.Mul(), loma_ir.ConstFloat(2.0), loma_ir.Call('sqrt',node.args)))
                stmts_left = self.mutate_expr(node.args[0])
                self.adjoint = old_adjoint
                return stmts_left

            if node.id == 'pow':
                old_adjoint = self.adjoint
                tmp1 = loma_ir.BinaryOp(loma_ir.Sub(), node.args[1], loma_ir.ConstFloat(1.0))
                tmp2 = loma_ir.Call('pow',[node.args[0], tmp1],lineno = node.lineno)
                tmp3 = loma_ir.BinaryOp(loma_ir.Mul(), node.args[1], tmp2)
                self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, tmp3)
                stmts_left = self.mutate_expr(node.args[0])
                self.adjoint = old_adjoint

                old_adjoint = self.adjoint
                tmp4 = loma_ir.Call('pow',[node.args[0], node.args[1]],lineno = node.lineno)
                tmp5 = loma_ir.BinaryOp(loma_ir.Mul(), tmp4 ,loma_ir.Call('log',[node.args[0]],lineno = node.lineno))
                self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, tmp5)
                stmts_right = self.mutate_expr(node.args[1])
                self.adjoint = old_adjoint

                return stmts_left + stmts_right
            
            if node.id == 'exp':
                old_adjoint = self.adjoint
                self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), self.adjoint, loma_ir.Call('exp',[node.args[0]],lineno = node.lineno))
                stmts_left = self.mutate_expr(node.args[0])
                self.adjoint = old_adjoint

                return stmts_left
            
            if node.id == 'log':
                old_adjoint = self.adjoint
                self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), self.adjoint, node.args[0])
                stmts_left = self.mutate_expr(node.args[0])
                self.adjoint = old_adjoint

                return stmts_left
            
            if node.id == 'int2float':
                old_adj = self.adjoint
                self.adjoint = loma_ir.ConstFloat(0.0)
                stmts = self.mutate_expr(node.args[0])
                self.adjoint = old_adj
                return stmts
            
            if node.id == 'float2int':
                old_adj = self.adjoint
                self.adjoint = loma_ir.ConstInt(0)
                stmts = self.mutate_expr(node.args[0])
                self.adjoint = old_adj
                return stmts
            
            return []


    return RevDiffMutator().mutate_function_def(func)



