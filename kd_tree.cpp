#ifndef KDTREE_INCLUDED
#define KDTREE_INCLUDED

#include <iostream>
#include "Point.h"
#include "BoundedPQueue.h"
#include <stdexcept>
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>

/*
/ Basic node used to build the KDTree
*/
template <typename TKey, typename TValue>
struct Node {
    TKey key = TKey();
    TValue value = TValue(); 
    Node* left = nullptr;
    Node* right = nullptr;
    int level = 0;
};

/*
/ Helper function that makes it easier to create KDTNodes
*/
template <typename TKey, typename TValue>
Node<TKey, TValue>* make_kdt_node(TKey key, TValue value) {
    Node<TKey, TValue>* node = new Node<TKey, TValue>;
    node->key = key;
    node->value = value;
    return node;
}


template <size_t Dim, typename T>
struct KDTree {
private: 
    Node<Point<Dim>, T>* root_node; //This node is our hook onto our KDTree

    //Builds our KDTree by recursively partitioning our dataset and splitting it into a left and right part.
    //The first node becomes the root to the left and right parts which we call the function with untill they
    //are empty.
    Node<Point<Dim>, T>*  build_tree(std::vector<std::pair<Point<Dim>*, T*> > dataSet, int d); 

    //Recursively searches the KDTree and finds the node with the correct key/Point<Dim>.
    Node<Point<Dim>, T>*  find(Point<Dim> target, Node<Point<Dim>, T>* node) const; 

    //Recursively deletes all nodes connected to the inputted note.
    void free_tree(Node<Point<Dim>, T>* node);
    
    //Recursively traverses all nodes in tree and returns 1 for every node found.
    int traverse(Node<Point<Dim>, T>* node) const;
    
    //Recursively traverses all nodes in tree and inserts in this tree.
    void copy_tree(Node<Point<Dim>, T>* node); 
public:
    // Constructor: KDTree();
    // Usage: kd_tree<3, int> tree_name;
    // ----------------------------------------------------
    // Constructs an empty tree.
    KDTree();
    
    // Destructor: ~KDTree()
    // Usage: Pliz no memoryleak mr Stroustrup :)
    // ----------------------------------------------------
    // Cleans up all resources used by the tree.
    ~KDTree();
    
    // KDTree(const KDTree& rhs);
    // KDTree& operator=(const KDTree& rhs);
    // Usage: KDTree<3, int> one = two;
    // Usage: one = two;
    // -----------------------------------------------------
    // Deep-copies the contents of another KDTree into this one.
    KDTree(const KDTree& rhs);
    KDTree& operator=(const KDTree& rhs);
    
    // size_t dimension() const;
    // Usage: size_t dim = kd.dimension();
    // ----------------------------------------------------
    // Returns the dimension of the points stored in this KDTree.
    size_t dimension() const;
    
    // size_t size() const;
    // bool empty() const;
    // Usage: if (kd.empty())
    // ----------------------------------------------------
    // Returns the number of elements in the kd-tree and whether the tree is
    // empty.
    size_t size() const;
    bool empty() const;
    
    // bool contains(const Point<Dim>& point) const;
    // Usage: if (kd.contains(point))
    // ----------------------------------------------------
    // Returns whether the specified point is contained in the KDTree.
    bool contains(const Point<Dim>& point) const;
    
    
    // void insert(const Point<Dim>& point, const T& value);
    // Usage: kd.insert(v, "This value is associated with v.");
    // ----------------------------------------------------
    // Inserts the point point into the KDTree, associating it with the specified
    // value. If the element already existed in the tree, the new value will
    // overwrite the existing one.
    void insert(const Point<Dim>& point, const T& value);
    
    // T& operator[](const Point<Dim>& point);
    // Usage: kd[v] = "Some Value";
    // ----------------------------------------------------
    // Returns a reference to the value associated with point point in the KDTree.
    // If the point does not exist, then it is added to the KDTree using the
    // default value of T as its key.
    T& operator[](const Point<Dim>& point);
    
    // T& at(const Point<Dim>& point);
    // const T& at(const Point<Dim>& point) const;
    // Usage: cout << kd.at(v) << endl;
    // ----------------------------------------------------
    // Returns a reference to the key associated with the point point. If the point
    // is not in the tree, this function throws an out_of_range exception.
    T& at(const Point<Dim>& point);
    const T& at(const Point<Dim>& point) const;
    
    // T kNNValue(const Point<Dim>& key, size_t k) const
    // Usage: cout << kd.kNNValue(v, 3) << endl;
    // ----------------------------------------------------
    // Given a point v and an integer k, finds the k points in the KDTree
    // nearest to v and returns the most common value associated with those
    // points. In the event of a tie, one of the most frequent value will be
    // chosen.
    T kNNValue(const Point<Dim>& key, size_t k) const;

    //kNNValue uses this helper function that uses a bpq to recursively search the tree and add the k nearest neighbors. 
    void get_neighbors(const Point<Dim> point, Node<Point<Dim>, T>* node, BoundedPQueue<Point<Dim> >& bpq) const; 


};

template <size_t Dim, typename T>
KDTree<Dim, T>::KDTree() {
    root_node = nullptr;
}

template <size_t Dim, typename T>
void KDTree<Dim, T>::copy_tree(Node<Point<Dim>, T>* node){
    if(node != nullptr) {
        insert(node->key, node->value);
        copy_tree(node->left);
        copy_tree(node->right);
    }
}

template <size_t Dim, typename T>
KDTree<Dim, T>::~KDTree() {
    free_tree(root_node);
}

template <size_t Dim, typename T>
KDTree<Dim, T>::KDTree(const KDTree& other) {
    root_node = nullptr;
    copy_tree(other.root_node);
}	

template <size_t Dim, typename T>
KDTree<Dim, T>& KDTree<Dim, T>::operator= (const KDTree& other) {
    root_node = nullptr;
    copy_tree(other.root_node);
    return *this;
}

template <size_t Dim, typename T>
void KDTree<Dim, T>::free_tree(Node<Point<Dim>, T>* node) {
    if(node != nullptr) {
        free_tree(node->right);
        free_tree(node->left);
        delete node;
    }
}

template <size_t Dim, typename T>
Node<Point<Dim>, T>*  KDTree<Dim, T>::find(Point<Dim> target, Node<Point<Dim>, T>* node) const{
    //Find now returns the last node it was at if it doesn't find the node we are searching for.
    if(empty()) 
        return nullptr;
    int level = node->level % Dim;
    if(node == nullptr)
        return nullptr;
    if(node->key == target)
        return node;
    if(target[level] >= node->key[level]) { 
        if(node->right != nullptr)
            return find(target, node->right);
        return node;
    }
    else { 
        if(node->left != nullptr)
            return find(target, node->left);
        return node;
    }
}

template <size_t Dim, typename T>
size_t KDTree<Dim, T>::dimension() const {
    return Dim;
}

template <size_t Dim, typename T>
int KDTree<Dim, T>::traverse(Node<Point<Dim>, T>* node) const {
    //This is new since we can nolonger use the vector for size.
    if(node != nullptr)
        return 1 + traverse(node->left) + traverse(node->right);
    return 0;
}

template <size_t Dim, typename T>
size_t KDTree<Dim, T>::size() const {
    return traverse(root_node);
}

template <size_t Dim, typename T>
bool KDTree<Dim, T>::empty() const {
    return root_node == nullptr;
}

template <size_t Dim, typename T>
void KDTree<Dim, T>::insert(const Point<Dim>& point, const T& value) {

    if(empty()) {
        root_node = make_kdt_node(point, value);
        root_node->level = 1;
    }
    else {
        //cout << root_node->value << endl;
        Node<Point<Dim>, T>* parrent = find(point, root_node);
        int level = parrent->level;
        if(parrent->key == point) {
            parrent->value = value;
        }
        else {
            Node<Point<Dim>, T>* child = make_kdt_node(point, value); 
            child->level = (level + 1) % Dim;
            if(child->key[level] >= parrent->key[level])
                parrent->right= child;
            else
                parrent->left = child;
        }
    }
}

template <size_t Dim, typename T>
bool KDTree<Dim, T>::contains(const Point<Dim>& point) const {
    return !empty() && find(point, root_node)->key == point; 
}

template <size_t Dim, typename T>
T& KDTree<Dim, T>::operator[] (const Point<Dim>& point) {

    Node<Point<Dim>, T>* node = find(point, root_node);
    if(!empty() && node->key == point)
        return node->value;
    else {
        insert(point, T());
        return find(point, root_node)->value;
    }
}

template <size_t Dim, typename T>
T& KDTree<Dim, T>::at(const Point<Dim>& point) {

    Node<Point<Dim>, T>* node = find(point, root_node);
    if(!empty() && node->key == point)
        return node->value;
    else
        throw std::out_of_range ("Key does not exist");

}

template <size_t Dim, typename T>
const T& KDTree<Dim, T>::at(const Point<Dim>& point) const {

    Node<Point<Dim>, T>* node = find(point, root_node);
    if(!empty() && node->key == point)
        return node->value;
    else
        throw std::out_of_range ("Key does not exist");
}

template <size_t Dim, typename T>
T KDTree<Dim, T>::kNNValue(const Point<Dim>& point, size_t k) const {

    BoundedPQueue<Point<Dim> > bpq(k); 
    get_neighbors(point, this->root_node, bpq);
    std::map<T, int> counter;
    std::vector<std::pair<T, int> > elemList;
    while(!bpq.empty()) {
       counter[at(bpq.dequeueMin())]++;
    }
    
    std::copy(counter.begin(), counter.end(), std::back_inserter<std::vector<std::pair<T, int> > >(elemList));
    std::sort(
        elemList.begin(), 
        elemList.end(), 
        [](const std::pair<T, int>& a, const std::pair<T, int>& b) { 
            return a.second < b.second; 
        }
    ); 

    return elemList.back().first;
}

template <size_t Dim, typename T>
void KDTree<Dim, T>::get_neighbors(const Point<Dim> point, Node<Point<Dim>, T>* node, BoundedPQueue<Point<Dim> >& bpq) const {
    
    if(node != nullptr) {
        bpq.enqueue(node->key, Distance(node->key, point));
        if(point[node->level] < node->key[node->level]) {
            get_neighbors(point, node->left, bpq);
            if(bpq.size() != bpq.maxSize() || abs(node->key[node->level] - point[node->level]) < bpq.worst())
                get_neighbors(point, node->right, bpq);
        }
        else if (point[node->level] >= node->key[node->level]) {
            get_neighbors(point, node->right, bpq);
            if(bpq.size() != bpq.maxSize() || abs(node->key[node->level] - point[node->level]) < bpq.worst())
                get_neighbors(point, node->left, bpq);
        }
    }
}

#endif 
