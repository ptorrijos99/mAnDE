package org.albacete.simd.mAnDE;

import java.util.ArrayList;
import java.util.HashMap;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public class Node {

    /**
     * ID of the Node.
     */
    private String id;
    
    /**
     * Name of the node.
     */
    private String name;
    
    /**
     * Name of the parent of the node.
     */
    private Node parent;
    
    /**
     * HashMap with the names of the children of the node.
     */
    private HashMap<String, Node> children;

    /**
     * Constructor. Creates a parent node, passing it the ID and name.
     *
     * @param id Node ID
     * @param name Name of the node
     */
    public Node(String id, String name) {
        this.parent = this;
        this.id = id;
        this.name = name;
        this.children = new HashMap();
    }

    /**
     * Constructor. Creates a child node, passing it the ID and parent node.
     *
     * @param id Node ID
     * @param parent Parent of the node
     */
    public Node(String id, Node parent) {
        this.parent = parent;
        this.id = id;
        this.name = "";
        this.children = new HashMap();
    }

    /**
     * Add a child.
     *
     * @param child Child to add
     */
    protected void addChild(Node child) {
        this.children.put(child.getId(), child);
    }

    /**
     * Returns the array of children.
     *
     * @return The array of children
     */
    protected ArrayList<String> getChildrenArray() {
        ArrayList<String> array = new ArrayList();
        children.values().forEach((child) -> {
            if (!child.name.equals("")) {
                array.add(child.name);
            }
        });
        return array;
    }

    /**
     * Returns the array of children that are not "no".
     *
     * @param no The child not to return.
     * @return The array of children
     */
    protected ArrayList<String> getChildrenArray(String no) {
        ArrayList<String> array = new ArrayList();
        children.values().forEach((child) -> {
            if ((!child.name.equals(no)) && (!child.name.equals(""))) {
                array.add(child.name);
            }
        });
        return array;
    }

    /**
     * @return the ID
     */
    public String getId() {
        return id;
    }

    /**
     * @param id the ID to set
     */
    public void setId(String id) {
        this.id = id;
    }

    /**
     * @return the parent
     */
    public Node getParent() {
        return parent;
    }

    /**
     * @param padre the parent to set
     */
    public void setParent(Node padre) {
        this.parent = padre;
    }

    /**
     * @return the children
     */
    public HashMap<String, Node> getChildren() {
        return children;
    }

    /**
     * @param children the children to set
     */
    public void setChildren(HashMap<String, Node> children) {
        this.children = children;
    }

    /**
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * @param name the name to set
     */
    public void setName(String name) {
        this.name = name;
    }
}
