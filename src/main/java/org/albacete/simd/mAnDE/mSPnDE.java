package org.albacete.simd.mAnDE;

import java.util.ArrayList;
import weka.core.Instance;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public interface mSPnDE {
    
    void buildTables();
    
    double[] probsForInstance(Instance inst);
    
    void moreChildren(String child);
    
    void moreChildren(ArrayList<String> children);
    
    int getNChildren();
    
    @Override
    boolean equals(Object o);
    
    @Override
    int hashCode();
}
