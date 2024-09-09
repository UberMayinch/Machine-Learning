#include <bits/stdc++.h>
using namespace std;

class node
{
public:
    double val;
    string op;
    double grad;
    multiset<shared_ptr<node>> children;

    // function pointer for something that takes in a node and returns a double (gradient value)
    double (*backward)(node);

    node(double x, multiset<shared_ptr<node>> children = {}, string oper = "", double gradient = 0, double(*backward)(node) = nullptr)
    {
        this->val = x;
        this->op = oper;
        this->grad = gradient;
        this->children = children;
        this->backward = backward;
    }

    // defined so that I can use multiset
    bool operator<(const node &A) const
    {
        return this->val < A.val;
    }
};

// Function to add two nodes and return the result
node addition(node &A, node &B)
{
    multiset<shared_ptr<node>> c;
    c.insert(make_shared<node>(A));
    c.insert(make_shared<node>(B));
    node C = node(A.val + B.val, c, "+");
    return C;
}

// Function to multiply two nodes and return the result
node multiply(node &A, node &B)
{
    multiset<shared_ptr<node>> c;
    c.insert(make_shared<node>(A));
    c.insert(make_shared<node>(B));
    node C = node(A.val * B.val, c, "*");
    return C;
}

// Function to print the details of a node
void printNode(node &A)
{
    cout << "Value: " << A.val << endl;
    cout << "Gradient: " << A.grad << endl;
    cout << "Operation: " << A.op << endl;
    cout << "Children:" << endl;
    for (auto it : A.children)
    {
        cout << it->val << " ";
    }
    cout << endl;
}

void topoSort(node &C, vector<node> &lst)
{
    for (auto it : C.children)
    {
        topoSort(*it, lst);
    }
    lst.push_back(C);
}

void BackProp(node& C){
    vector<node> lst;
    topoSort(C,lst); 
    reverse(lst.begin(), lst.end());
    C.grad = 1;
    for(auto it: lst){

        
    }
}

int main()
{
    node A(13);
    node B(14);
    node C = multiply(A, B);
    
    
    vector<node> lst;
    topoSort(C, lst);
    
    // Print the sorted nodes
    reverse(lst.begin(), lst.end());
    for (auto &it : lst)
    {
        printNode(it);
    }
}
